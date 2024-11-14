import numpy as np
import spacy
import pickle
import datetime
import json
from dataloader import read_data

nlp = spacy.load('en_core_web_lg')

def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet


def dependency_adj_matrix(text, senticNet):
    # https://spacy.io/docs/usage/processing-text
    text = text.lower()
    document = nlp(text)
    seq_len = len(document)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    #print('='*20+':')
    #print(document)
    #print(senticNet)

    tokens = []
    for token in document:
        #print('token:', token)
        tokens.append(str(token))
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) + 1
        else:
            sentic = 0
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic

    return matrix, tokens


def process(filename):
    senticNet = load_sentic_word()
    sentences_list, truth_pairs_list = read_data(filename)

    idx2graph = {}
    sent_tokens_list = []
    for i, sent in enumerate(sentences_list):
        adj_matrix, tokens = dependency_adj_matrix(sent, senticNet)
        idx2graph[i] = adj_matrix
        sent_tokens_list.append(tokens)

    fout = open(filename+'.graph', 'wb')
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
    fout.close()

    processed_data = []
    for sent_tokens, truth_pairs in zip(sent_tokens_list, truth_pairs_list):
        truth_pairs = set(truth_pairs)
        processed_data.append({
            'sentence': ' '.join(sent_tokens),
            'truth_pairs': list(truth_pairs)
        })
    with open(filename+'.processed', 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == '__main__':
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/train_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/dev_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/test_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/dev_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/train_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/dev_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/test_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/train_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/dev_triplets.txt')
    process('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/test_triplets.txt')
    process('SemEval-Triplet-data/Integrated-Splits/dev_triplets.txt')
    process('SemEval-Triplet-data/Integrated-Splits/test_triplets.txt')
    process('SemEval-Triplet-data/Integrated-Splits/train_triplets.txt')


