from torch.utils.data import Dataset
import pandas as pd
import json
import torch
from transformers import RobertaModel, RobertaTokenizer
import sys
# sys.path.append('./utils')
from utils.config import get_config
config = get_config()
from utils.transition_action import get_action_sequence
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

sent2id = {'POS': 0, 'NEU': 1, 'NEG': 2}


class TripletDataset(Dataset):
    
    def __init__(self, sentences_list, truth_pairs_list, graphs, mode):
        self.tokenizer = RobertaTokenizer.from_pretrained(config.roberta_path)
        
        truth_pairs_for_bert_list = []
        sent_token_ids_list = []
        sent_tokens_list = []
        new_graph_list = []
        for sent_text, truth_pairs, graph in zip(sentences_list, truth_pairs_list, graphs.values()):
            new_row_graph = []
            orig_pos2bert_pos = {}
            sent_tokens = sent_text.split(' ')
            sent_tokens_for_bert = []
            for orig_pos, (token, row) in enumerate(zip(sent_tokens, graph)):
                bert_tokens = self.tokenizer.tokenize(token)
                cur_len = len(sent_tokens_for_bert)

                orig_pos2bert_pos[orig_pos] = (cur_len, cur_len+len(bert_tokens)-1)
                sent_tokens_for_bert += bert_tokens
                for t in bert_tokens:
                    new_row_graph.append(row)
            new_row_graph = np.array(new_row_graph)

            new_graph = []
            for orig_pos, (token, row) in enumerate(zip(sent_tokens, new_row_graph.transpose())):
                bert_tokens = self.tokenizer.tokenize(token)
                for t in bert_tokens:
                    new_graph.append(row)
            new_graph = np.array(new_graph).transpose()

            assert new_graph.shape[0] == new_graph.shape[1]
            assert np.array_equal(new_graph, new_graph.T)
            assert new_graph.shape[0] == len(sent_tokens_for_bert)

            new_graph_list.append(new_graph)

            sent_tokens_list.append(sent_tokens_for_bert)
            sent_tokens_for_bert = ['[CLS]'] + sent_tokens_for_bert + ['[SEP]']
            sent_token_ids_list.append(self.tokenizer.convert_tokens_to_ids(sent_tokens_for_bert))
            
            truth_pairs_for_bert = []
            for pair in truth_pairs:
                source_span = pair[0]
                target_span = pair[1]
                source_span_for_bert = (orig_pos2bert_pos[source_span[0]][0], orig_pos2bert_pos[source_span[1]][1])
                target_span_for_bert = (orig_pos2bert_pos[target_span[0]][0], orig_pos2bert_pos[target_span[1]][1])
                truth_pairs_for_bert.append((source_span_for_bert, target_span_for_bert, sent2id[pair[2]]))

            truth_pairs_for_bert_list.append(truth_pairs_for_bert)
        self.truth_pairs_list = truth_pairs_for_bert_list
        self.sent_token_ids_list = sent_token_ids_list
        self.sent_tokens_list = sent_tokens_list
        self.graph_list = new_graph_list

        
        parser_states_list = []
        for sent_tokens, truth_pairs in zip(sent_tokens_list, truth_pairs_for_bert_list):
            truth_pairs = set(truth_pairs)
            actions, _ = get_action_sequence(sent_tokens, truth_pairs)
            parser_states_list.append(actions)
        self.parser_states_list = parser_states_list
        

    def __len__(self):
        return len(self.sent_token_ids_list)

    def __getitem__(self, index):
        one_sample = (
            self.sent_tokens_list[index],
            self.sent_token_ids_list[index],
            self.truth_pairs_list[index],
            self.parser_states_list[index],
            self.graph_list[index]
        )
        return one_sample

def generate_batch_fn(batch):
    batch = list(zip(*batch))
    batch = {
        'sent_tokens_list': batch[0],
        'sent_token_ids_list': batch[1],
        'truth_pairs_list': batch[2],
        'parser_states_list': batch[3],
        'graph_list': batch[4]
    }
    return batch

def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    truth_pairs_list = []
    sentences_list = []
    for line in lines:
        sentence, triplets = line.split('####')
        triplets = eval(triplets)
        pair_tuple_set = set()
        for triplet in triplets:
            if len(triplet[0]) == 1: 
                aspect = triplet[0] * 2
            else:
                aspect = triplet[0]
            if len(triplet[1]) == 1: 
                opinion = triplet[1] * 2
            else:
                opinion = triplet[1]

            if len(aspect) > 2:
                aspect = (aspect[0], aspect[-1])
            if len(opinion) > 2:
                opinion = (triplet[1][0], triplet[1][-1])
            
            sentiment = triplet[2]
            pair_tuple_set.add((tuple(aspect), tuple(opinion), sentiment))
        truth_pairs_list.append(pair_tuple_set)
        sentences_list.append(sentence)
    return sentences_list, truth_pairs_list

if __name__ == '__main__':

    dataset_path = 'SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/train_triplets.txt'
    sentences_list, truth_pairs_list = read_data(dataset_path)

    mode = 'train'
    dataset = TripletDataset(sentences_list, truth_pairs_list, mode)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=generate_batch_fn)
    for batch in dataloader:
        print(batch)
        break