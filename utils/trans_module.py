import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('./utils')
from transformers import BertModel, BertTokenizer
from transformers import DebertaModel, DebertaTokenizer
from copy import deepcopy
import numpy as np
from copy import deepcopy
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import os
from transformers import RobertaModel, RobertaTokenizer


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class RobertaEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(config.roberta_path)
        self.gc1 = GraphConvolution(config.roberta_output_size, config.roberta_output_size)
        self.gc2 = GraphConvolution(config.roberta_output_size, config.roberta_output_size)
        self.gc3 = GraphConvolution(config.roberta_output_size, config.roberta_output_size)
        self.tokenizer = RobertaTokenizer.from_pretrained(config.roberta_path)
        self.max_AC_num = config.max_AC_num
        self.dropout = config.dropout
        self.config = config
        self.linear = nn.Linear(config.roberta_output_size, config.mlp_size)
        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(normalized_shape=config.roberta_output_size, eps=config.layer_norm_eps)

    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list

    def forward(self, sent_token_ids_list, graph_list):
        ids_padding_list, mask_list = self.padding_and_mask(sent_token_ids_list)
        max_seq_len = len(ids_padding_list[0])
        padded_graph_list = []
        for graph in graph_list:
            graph = np.pad(graph, ((0, max_seq_len - graph.shape[0]), (0, max_seq_len - graph.shape[0])), 'constant')
            padded_graph_list.append(graph)
        padded_graph_array = np.array(padded_graph_list)
        ids_padding_tensor = torch.LongTensor(ids_padding_list).cuda()
        adj = torch.tensor(padded_graph_array).cuda()
        mask_tensor = torch.tensor(mask_list).cuda()
        roberta_outputs = self.roberta(ids_padding_tensor, attention_mask=mask_tensor)
        x = roberta_outputs[0]
        if self.config.gcn_layer_num >= 1:
            x = self.activation(self.gc1(x, adj))
        if self.config.gcn_layer_num >= 2:
            x = self.activation(self.gc2(x, adj))
        if self.config.gcn_layer_num >= 3:
            x = self.activation(self.gc3(x, adj))
        if self.config.gcn_add_roberta_output:
            x = x + roberta_outputs[0]

        if self.config.attn:
            alpha_mat = torch.matmul(x, roberta_outputs[0].transpose(1, 2))
            alpha = F.softmax(alpha_mat, dim=2)
            x = torch.matmul(alpha, roberta_outputs[0])

        word_reps_list = [x[batch_i, 1:len(sent_token_ids) - 1, :] for batch_i, sent_token_ids in
                          enumerate(sent_token_ids_list)]
        return word_reps_list

    def get_onehot_position_info(self, forward_pos_list, backward_pos_list, para_type_list):
        batch_size = len(forward_pos_list)

        def pos2onehot(pos, max_AC_num):
            pos = pos % max_AC_num
            pos_onehot = torch.zeros(pos.shape[0], max_AC_num)
            pos = pos.unsqueeze(-1)
            pos_onehot = pos_onehot.scatter_(1, pos.cpu(), 1).cuda()
            return pos_onehot

        for_onehot_pos_list = list(map(pos2onehot, forward_pos_list, [self.max_AC_num] * batch_size))
        back_onehot_pos_list = list(map(pos2onehot, backward_pos_list, [self.max_AC_num] * batch_size))
        para_onehot_pos_list = list(map(pos2onehot, para_type_list, [self.max_AC_num] * batch_size))
        return for_onehot_pos_list, back_onehot_pos_list, para_onehot_pos_list



class TransitionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.is_bi = config.is_bi
        self.bert_output_size = config.roberta_output_size
        self.mlp_size = config.mlp_size
        self.cell_size = config.cell_size
        self.scale_factor = config.scale_factor
        self.dropout = config.dropout
        self.lstm_layers = config.lstm_layers
        self.max_AC_num = config.max_AC_num
        self.max_dist_len = config.max_dist_len
        self.position_ebd_dim = config.position_ebd_dim

        self.distance_embedding = nn.Embedding(self.max_dist_len, self.position_ebd_dim)
        self.position_trainable = config.position_trainable
        self.action_ebd_dim = config.action_ebd_dim
        self.action_type_num = config.action_type_num
        self.action_embedding = nn.Embedding(self.action_type_num, self.action_ebd_dim)
        self.action_trainable = config.action_trainable
        self.action_label_num = config.action_label_num
        self.AC_type_label_num = config.AC_type_label_num
        self.dataset_name = config.dataset_name
        self.AC_type_input_size = self.bert_output_size * 2
        self.rel_type_input_size = self.bert_output_size * 2
        self.lstm_input_size = self.bert_output_size

        self.stack_lstm = nn.LSTM(self.lstm_input_size, self.cell_size, self.lstm_layers, bidirectional=self.is_bi)
        self.buffer_lstm = nn.LSTM(self.lstm_input_size, self.cell_size, self.lstm_layers, bidirectional=self.is_bi)
        self.action_lstm = nn.LSTM(self.action_ebd_dim, self.cell_size, self.lstm_layers, bidirectional=False)
        self.trans_input_size = self.bert_output_size * 2 + self.action_ebd_dim + self.position_ebd_dim
        intermediate_size = self.mlp_size // 2
        self.action_MLP = nn.Sequential(
            nn.Linear(self.trans_input_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, self.mlp_size // self.scale_factor),
            nn.BatchNorm1d(self.mlp_size // self.scale_factor),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size // self.scale_factor, self.action_label_num)
        )
        self.sent_mlp = nn.Sequential(
            nn.Linear(self.trans_input_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, self.mlp_size // self.scale_factor),
            nn.BatchNorm1d(self.mlp_size // self.scale_factor),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size // self.scale_factor, self.AC_type_label_num)
        )

        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                if len(param.data.size()) > 1:
                    nn.init.xavier_normal_(param.data)
                else:
                    param.data.uniform_(-0.1, 0.1)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue
        self.distance_embedding.weight.requires_grad = self.position_trainable
        self.action_embedding.weight.requires_grad = self.action_trainable

    def action_encoder(self, parser_state_list):
        action_list = [x[2] for asl in parser_state_list for x in asl]
        action_len_list = [len(x) for x in action_list]
        max_action_len = max(action_len_list)
        action_padding_list = [[6] + x[:-1] + [7] * (max_action_len - len(x)) \
                                   if len(x) != 0 else [6] + [7] * (max_action_len - 1) \
                               for x in action_list]
        action_padding_tensor = torch.tensor(action_padding_list).cuda()

        inputs = self.action_embedding(action_padding_tensor).permute(1, 0, 2)
        bs = inputs.size()[1]
        outputs, _ = self.action_lstm(inputs)
        outputs_permute = outputs.permute(1, 0, 2)
        output_list = [outputs_permute[i][al - 1] for i, al in enumerate(action_len_list)]
        output_stack = torch.stack(output_list)
        return output_stack

        def contrastive_loss(self, action_logits, true_action_tensor):
        action_probs = F.softmax(action_logits, dim=1)
        pred_action = action_probs.argmax(dim=1)

        pred_action_embed = self.action_embedding(pred_action)  # [batch_size, action_ebd_dim]
        true_action_embed = self.action_embedding(true_action_tensor)  # [batch_size, action_ebd_dim]

        sim_matrix = F.cosine_similarity(
            pred_action_embed.unsqueeze(1),  # [batch_size, 1, action_ebd_dim]
            true_action_embed.unsqueeze(0),  # [1, batch_size, action_ebd_dim]
            dim=2
        )

        positive_mask = torch.eye(sim_matrix.size(0)).cuda()
        negative_mask = 1 - positive_mask

        exp_sim = torch.exp(sim_matrix)

        pos_sim = (exp_sim * positive_mask).sum(dim=1)

        all_sim = (exp_sim * (positive_mask + negative_mask)).sum(dim=1)

        loss = -torch.log(pos_sim / all_sim)

        return loss.mean()


    def train_mode(self, word_reps_list, parser_state_list):
        true_action_list, distance_list = [], []
        sk_reps_list, bf_reps_list = [], []
        true_sent_list = []

        for word_reps, parser_state in zip(word_reps_list, parser_state_list):
            for state in parser_state:
                sk, bf, all_actions, sent_label = state[0], state[1], state[2], state[3]
                cur_action = all_actions[-1]
                stack_reps = []
                for s in sk:
                    term_words_reps = word_reps[s[0]:s[1] + 1]
                    term_rep = torch.mean(term_words_reps, 0)
                    stack_reps.append(term_rep)
                stack_reps = torch.stack(stack_reps)

                sk_reps_list.append(stack_reps)
                distance_list.append(int(abs(sk[-2][0] - sk[-1][0])))

                if len(bf) > 0:
                    bf_reps = torch.stack([word_reps[b] for b in bf])
                else:
                    bf_reps = stack_reps[-1].unsqueeze(0)
                bf_reps_list.append(bf_reps)
                true_action_list.append(cur_action)
                true_sent_list.append(sent_label)

        # Prepare embeddings for contrastive loss
        anchor = sk_reps_list[0]  # Example anchor
        positive = sk_reps_list[1]  # Example positive with similar sentiment
        negative = bf_reps_list[0]  # Example negative with different sentiment
        # print("Anchor shape:", anchor.shape)
        # print("Positive shape:", positive.shape)
        # print("Negative shape:", negative.shape)

        # contrastive_loss_value = self.contrastive_loss(anchor, positive, negative)
        #
        # contrastive_loss_value = 0
        # for label in torch.unique(true_sent_tensor):
        #     mask = (true_sent_tensor == label)
        #     if torch.sum(mask) > 1:  # Need at least 2 samples
        #         label_reps = stack_reps[mask]
        #         contrastive_loss_value += self.contrastive_loss(label_reps, label_reps)

        sk_reps_packed = pack_sequence(sk_reps_list, enforce_sorted=False)
        bf_reps_packed = pack_sequence(bf_reps_list, enforce_sorted=False)
        sk_lstm_out_packed, _ = self.stack_lstm(sk_reps_packed)
        bf_lstm_out_packed, _ = self.buffer_lstm(bf_reps_packed)
        sk_lstm_out_padded, sk_len_tensor = pad_packed_sequence(sk_lstm_out_packed, batch_first=True)
        bf_lstm_out_padded, bf_len_tensor = pad_packed_sequence(bf_lstm_out_packed, batch_first=True)
        sk_reps_list = [sk_lstm_out_padded[i][:sk_len] for i, sk_len in enumerate(sk_len_tensor)]
        bf_reps_list = [bf_lstm_out_padded[i][:bf_len] for i, bf_len in enumerate(bf_len_tensor)]
        hist_action_tensor = self.action_encoder(parser_state_list)

        state_reps_list = []
        for sk_reps, bf_reps in zip(sk_reps_list, bf_reps_list):
            state_reps = torch.cat([sk_reps[-2], sk_reps[-1], bf_reps[0]])
            state_reps_list.append(state_reps)

        distance_tensor = torch.tensor(distance_list).cuda()
        distance_embedding = self.distance_embedding(distance_tensor)
        final_feat_tensor = torch.cat([torch.stack(state_reps_list),
                                       distance_embedding,
                                       hist_action_tensor], 1)
        true_action_tensor = torch.LongTensor(true_action_list).cuda()
        true_sent_tensor = torch.LongTensor(true_sent_list).cuda()
        action_logits = self.action_MLP(final_feat_tensor)
        sent_logits = self.sent_mlp(final_feat_tensor)

        contrastive_loss_value = self.contrastive_loss(action_logits, true_action_tensor)
        contrastive_loss_value_senti = self.contrastive_loss(sent_logits, true_sent_tensor)

        action_loss = F.cross_entropy(action_logits, true_action_tensor)
        sentiment_loss = F.cross_entropy(sent_logits, true_sent_tensor)

        beta_cl = 0.1
        total_loss = action_loss + sentiment_loss + beta_cl * (contrastive_loss_value+contrastive_loss_value_senti)

        return total_loss, action_logits, sent_logits, true_action_tensor, true_sent_tensor

    def predict_action(self, word_reps, sk, bf, action_list):

        stack_reps = []
        for s in sk:
            term_words_reps = word_reps[s[0]:s[1] + 1]
            term_rep = torch.mean(term_words_reps, 0)
            stack_reps.append(term_rep)
        stack_reps = torch.stack(stack_reps).unsqueeze(0)

        if len(bf) > 0:
            bf_reps = torch.stack([word_reps[b] for b in bf]).unsqueeze(0)
        else:
            # if sk[0] < word_reps.size()[0]-1:
            #     bf_reps = torch.stack([word_reps[sk[0]+1]]).unsqueeze(0)
            # else:
            #     bf_reps = torch.stack([word_reps[sk[0]]]).unsqueeze(0)
            bf_reps = stack_reps[0][-1].unsqueeze(0).unsqueeze(0)

        act_reps = self.action_embedding(torch.tensor([action_list]).cuda())

        sk_lstm_out, _ = self.stack_lstm(stack_reps.permute(1, 0, 2))
        stack_reps = sk_lstm_out.permute(1, 0, 2).squeeze(0)

        bf_lstm_out, _ = self.buffer_lstm(bf_reps.permute(1, 0, 2))
        bf_reps = bf_lstm_out.permute(1, 0, 2).squeeze(0)

        act_lstm_out, act_hidden = self.action_lstm(act_reps)
        act_reps = act_lstm_out.squeeze(0).squeeze(0)[-1]

        state_reps = torch.cat([stack_reps[-2], stack_reps[-1], bf_reps[0]])
        distance = torch.tensor(int(abs(sk[-2][0] - sk[-1][0]))).cuda()
        distance_emb = self.distance_embedding(distance)

        final_feat_tensor = torch.cat([state_reps, distance_emb, act_reps]).unsqueeze(0)

        action_logits = self.action_MLP(final_feat_tensor)
        action_probs = F.softmax(action_logits, 1)
        pred_action = action_probs.argmax(1).data.cpu().numpy()[0]
        sent_logits = self.sent_mlp(final_feat_tensor)
        sent_probs = F.softmax(sent_logits, 1)
        pred_sent = sent_probs.argmax(1).data.cpu().numpy()[0]

        return pred_action, pred_sent

    def eval_mode(self, word_reps_list):
        action2id = {'shift': 0, 'lr': 1, 'rr': 2,
                     'lr_n': 3, 'rr_n': 4, 'merge': 5}
        id2action = {v: k for k, v in action2id.items()}
        pred_states_list = []
        pred_pairs_list = []

        for word_reps in word_reps_list:
            stack = []
            buffer = list(range(0, len(word_reps)))
            stack.append([0, 0]), stack.append([1, 1])
            buffer.remove(0), buffer.remove(1)
            actions = ['shift', 'shift']
            pairs = []
            states = []

            act_hidden = None
            action_list = [6, 0, 0]
            while len(buffer) > 0:
                if len(stack) < 2:
                    first_ele = buffer.pop(0)
                    stack.append([first_ele, first_ele])
                    actions.append('shift')
                s_pair = (tuple(stack[-2]), tuple(stack[-1]))

                action, pred_sent = self.predict_action(word_reps, stack, buffer, action_list)
                action_list.append(action)
                actions.append(id2action[action])
                states.append((deepcopy(stack), deepcopy(buffer), deepcopy(actions), pred_sent))

                pair = None
                if action == action2id['lr_n']:
                    stack.pop(-2)
                elif action == action2id['rr_n']:
                    stack.pop(-1)
                elif action == action2id['merge']:
                    end = stack.pop(-1)
                    start = stack.pop(-1)
                    stack.append([start[0], end[1]])
                    # assert len(stack) == 1
                elif action == action2id['rr']:
                    pair = (stack[-2], stack[-1])
                    stack.pop(-1)
                elif action == action2id['lr']:
                    pair = (stack[-1], stack[-2])
                    stack.pop(-1)
                elif action == action2id['shift']:
                    if len(buffer) > 0:
                        first_ele = buffer.pop(0)
                        stack.append([first_ele, first_ele])

                if pair is not None:
                    pairs.append([pair[0], pair[1], int(pred_sent)])

            # assert len(stack) <= 2

            while len(stack) > 1:
                s_pair = (tuple(stack[-2]), tuple(stack[-1]))
                action, pred_sent = self.predict_action(word_reps, stack, buffer, action_list)
                action_list.append(action)
                actions.append(id2action[action])
                states.append((deepcopy(stack), deepcopy(buffer), deepcopy(actions), pred_sent))

                pair = None
                if action == action2id['lr_n']:
                    stack.pop(-2)
                elif action == action2id['rr_n']:
                    stack.pop(-1)
                elif action == action2id['merge']:
                    end = stack.pop(-1)
                    start = stack.pop(-1)
                    stack.append([start[0], end[1]])
                    # assert len(stack) == 1
                elif action == action2id['rr']:
                    pair = (stack[-2], stack[-1])
                    stack.pop(-1)
                elif action == action2id['lr']:
                    pair = (stack[-1], stack[-2])
                    stack.pop(-2)
                elif action == action2id['shift']:
                    stack.pop(-1)

                if pair is not None:
                    pairs.append([pair[0], pair[1], int(pred_sent)])

            pred_states_list.append(states)
            pred_pairs_list.append(pairs)

        return pred_pairs_list, pred_states_list

    def forward(self, word_reps_list, parser_state_list, mode):
        if mode == 'train':
            total_loss, action_logits, sent_logits, true_action_tensor, true_sent_tensor = \
                self.train_mode(word_reps_list, parser_state_list)
            return total_loss, action_logits, sent_logits, true_action_tensor, true_sent_tensor
        elif mode == 'eval':
            pred_pairs_list, pred_states_list = \
                self.eval_mode(word_reps_list)
            return pred_pairs_list, pred_states_list
        else:
            print('mode error!')
