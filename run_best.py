import torch
import os
import json
import pickle
from torch.utils.data import DataLoader
from utils.trans_module import TransitionModel, DebertaEncoder
from utils.evaluation import evaluate
from dataloader import TripletDataset, generate_batch_fn
from utils.config import get_config


def load_model(model_path, config):
    """ Load the pre-trained model """
    base_encoder = DebertaEncoder(config)
    trans_model = TransitionModel(config)

    # Load model parameters
    trans_model.load_state_dict(torch.load(model_path))

    # Move models to GPU
    base_encoder.cuda()
    trans_model.cuda()

    return base_encoder, trans_model


def load_test_data(test_data_paths):
    """ Load test data from processed files """
    datasets = []
    for path in test_data_paths:
        with open(path, 'r') as f:
            loaded_data_list = json.load(f)
            test_sentences_list, test_truth_pairs_list = zip(
                *[(d['sentence'], d['truth_pairs']) for d in loaded_data_list])

        # Load corresponding graph
        test_graph_path = '.'.join(path.split('.')[:-1]) + '.graph'
        test_graphs = pickle.load(open(test_graph_path, 'rb'))
        datasets.append((test_sentences_list, test_truth_pairs_list, test_graphs))

    return datasets


def evaluate_model(model_path, test_data_paths, config, default_batch_size=4):
    """ Load the model and evaluate it on multiple test datasets """
    base_encoder, trans_model = load_model(model_path, config)

    batch_size = getattr(config, 'batch_size',
                         default_batch_size)  # Use config.batch_size if it exists, otherwise default to 4

    for test_data_path in test_data_paths:
        test_sentences_list, test_truth_pairs_list, test_graphs = load_test_data([test_data_path])[0]
        test_dataset = TripletDataset(test_sentences_list, test_truth_pairs_list, test_graphs, 'test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=generate_batch_fn)

        # Evaluate
        test_eval_res, _ = evaluate(trans_model, base_encoder, test_dataset, test_loader, 'eval')
        print(f"Results for {test_data_path}:")
        print(
            f"F1: {test_eval_res['f1']:.6f}, Recall: {test_eval_res['rec']:.6f}, Precision: {test_eval_res['pre']:.6f}\n")


if __name__ == "__main__":
    config = get_config()  # Load configuration

    # Paths
    model_path = "saved_models/20241111_155634_00111/trans_best.mdl"
    test_data_paths = [
        "SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/test_triplets.txt.processed",
        "SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt.processed",
        "SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/test_triplets.txt.processed",
        "SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/test_triplets.txt.processed"
    ]

    evaluate_model(model_path, test_data_paths, config)
