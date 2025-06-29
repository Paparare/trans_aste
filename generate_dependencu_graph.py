import sys
import numpy as np
import spacy
import pickle
import json
from dataloader import read_data

# Load the spaCy model once up-front
nlp = spacy.load("en_core_web_lg")


def dependency_adj_matrix(text: str):
    """
    Build an adjacency matrix from spaCy’s dependency parse.
    Each token has a self-loop (diagonal = 1) and an undirected edge
    to every syntactic child (weight = 1).  No SenticNet weighting.
    """
    text = text.lower()
    doc = nlp(text)
    seq_len = len(doc)
    matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

    tokens = [token.text for token in doc]

    for token in doc:
        if token.i < seq_len:
            # self-loop
            matrix[token.i, token.i] = 1.0
            # undirected edges to children
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i, child.i] = 1.0
                    matrix[child.i, token.i] = 1.0
    return matrix, tokens


def process(filename: str):
    """
    Parse a dataset file, build adjacency matrices for each sentence,
    and save the graphs and lightly-processed data.
    """
    sentences, truth_pairs_list = read_data(filename)

    idx2graph = {}
    sent_tokens_list = []

    for idx, sent in enumerate(sentences):
        adj_matrix, tokens = dependency_adj_matrix(sent)
        idx2graph[idx] = adj_matrix
        sent_tokens_list.append(tokens)

    with open(f"{filename}.graph", "wb") as fout:
        pickle.dump(idx2graph, fout)
    print("done !!!", filename)

    processed_data = []
    for tokens, truth_pairs in zip(sent_tokens_list, truth_pairs_list):
        processed_data.append(
            {
                "sentence": " ".join(tokens),
                "truth_pairs": list(set(truth_pairs)),
            }
        )

    with open(f"{filename}.processed", "w") as f:
        json.dump(processed_data, f, indent=4)


if __name__ == "__main__":
    # Example calls (uncomment / adjust as needed)
    # process("SemEval-Triplet-data/Integrated-Splits/merged1_14r15r16rtriplets_test.txt")
    # process("SemEval-Triplet-data/Integrated-Splits/merged1_14r15r16rtriplets.txt")
    process("SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/test_triplets.txt")
    process("SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt")
    process("SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/test_triplets.txt")
    process("SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/test_triplets.txt")


