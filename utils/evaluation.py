import sys
sys.path.append('../')
from sklearn.metrics import f1_score
import torch
from config import get_config
import itertools
config = get_config()

def get_eval_result(res_dict):
    res_msg = 'F1: {:.4f}\tPre: {:.4f}\tRec: {:.4f}'.format( \
            res_dict["f1"], res_dict["pre"], res_dict["rec"])
    return res_msg

def pair_metric(preds, grounds):
    tn, fn, fp, tp = 0, 0, 0, 0
    for i in range(len(preds)):
        pred, ground = preds[i], grounds[i]
        # Ensure elements are hashable by converting lists to tuples
        t_pair = set([(tuple(x[0]), tuple(x[1])) for x in ground])
        p_pair = set([(tuple(x[0]), tuple(x[1])) for x in pred])

        tp += len(p_pair & t_pair)
        fn += (len(t_pair) - len(p_pair & t_pair))
        fp += (len(p_pair) - len(p_pair & t_pair))

    pre = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = (2 * pre * rec) / (pre + rec + 1e-10)

    return pre, rec, f1


def evaluate(trans_model, base_encoder, dev_dataset, data_loader, mode="eval"):
    trans_model.eval()
    base_encoder.eval()
    all_pred_pairs, all_true_pairs = [], []

    for batch in data_loader:
        sent_tokens_list = batch['sent_tokens_list']
        sent_token_ids_list = batch['sent_token_ids_list']
        truth_pairs_list = batch['truth_pairs_list']
        parser_state_list = batch['parser_states_list']
        graph_list = batch['graph_list']

        word_reps_list = base_encoder(sent_token_ids_list, graph_list)
        pred_pairs_list, pred_states_list = trans_model(word_reps_list, parser_state_list, 'eval')

        all_pred_pairs.extend(pred_pairs_list)
        all_true_pairs.extend(truth_pairs_list)

    # Ignore the sentiment ID by considering only the first two elements of each pair
    all_pred_pairs = [[(x[0], x[1]) for x in pair] for pair in all_pred_pairs]
    all_true_pairs = [[(x[0], x[1]) for x in pair] for pair in all_true_pairs]

    pr_eval_res = pair_metric(all_pred_pairs, all_true_pairs)

    eval_res = {}
    eval_res["pre"] = pr_eval_res[0]
    eval_res["rec"] = pr_eval_res[1]
    eval_res["f1"] = pr_eval_res[2]
    return eval_res, all_pred_pairs
