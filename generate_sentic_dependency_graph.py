from pathlib import Path
import requests
import sys
import os
import json
import pickle
import datetime
import numpy as np
import spacy

from dataloader import read_data   
nlp = spacy.load("en_core_web_lg")

BASE_LOCAL = Path("SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020")
REPO_RAW   = (
    "https://raw.githubusercontent.com/"
    "xuuuluuu/Position-Aware-Tagging-for-ASTE/master/data/ASTE-Data-V2"
)
SPLITS     = ["14lap", "14res", "15res", "16res"]
FILES      = ["train_triplets.txt", "dev_triplets.txt", "test_triplets.txt"]
COMBINED_TRAIN = BASE_LOCAL / "combined_train.txt"
CHUNK      = 1024 * 16  

def download_file(url: str, dst: Path, verbose: bool = True) -> None:
    if dst.exists():
        if verbose:
            print(f"[skip] {dst} already exists")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=15) as r:
            r.raise_for_status()
            with dst.open("wb") as f:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
        if verbose:
            print(f"[ok]   downloaded → {dst}")
    except requests.RequestException as exc:
        print(f"[ERR]  failed to download {url} – {exc}")
        sys.exit(1)

def fetch_all() -> None:
    print("### Downloading ASTE-Data-V2 files")
    for split in SPLITS:
        for fname in FILES:
            url = f"{REPO_RAW}/{split}/{fname}"
            local_path = BASE_LOCAL / split / fname
            download_file(url, local_path)
            
def combine_train() -> None:
    print("\n### Combining training files into", COMBINED_TRAIN)
    with COMBINED_TRAIN.open("w", encoding="utf8") as fout:
        total = 0
        for split in SPLITS:
            src = BASE_LOCAL / split / "train_triplets.txt"
            lines = src.read_text(encoding="utf8").splitlines()
            fout.write("\n".join(lines) + "\n")
            total += len(lines)
            print(f"  appended {len(lines):>5} lines from {split}/train_triplets.txt")
    print(f"  → combined total: {total} lines\n")
    
def load_sentic_word():
    path = "./senticNet/senticnet_word.txt"
    senticNet = {}
    with open(path, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            word, sentic = line.split("\t")
            senticNet[word] = sentic
    return senticNet

def dependency_adj_matrix(text, senticNet):
    text = text.lower()
    document = nlp(text)
    seq_len = len(document)
    matrix = np.zeros((seq_len, seq_len), dtype="float32")
    tokens = []
    for token in document:
        tokens.append(str(token))
        sentic = float(senticNet.get(str(token), 0)) + 1
        if token.i < seq_len:
            matrix[token.i][token.i] = sentic
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = sentic
                    matrix[child.i][token.i] = sentic
    return matrix, tokens

def process(filename: str):
    senticNet = load_sentic_word()
    sentences_list, truth_pairs_list = read_data(filename)

    idx2graph = {}
    sent_tokens_list = []
    for i, sent in enumerate(sentences_list):
        adj_matrix, tokens = dependency_adj_matrix(sent, senticNet)
        idx2graph[i] = adj_matrix
        sent_tokens_list.append(tokens)

    with open(filename + ".graph", "wb") as fout:
        pickle.dump(idx2graph, fout)
    print("  ✓ graph saved ->", filename + ".graph")

    processed_data = []
    for sent_tokens, truth_pairs in zip(sent_tokens_list, truth_pairs_list):
        processed_data.append(
            {"sentence": " ".join(sent_tokens), "truth_pairs": list(set(truth_pairs))}
        )
    with open(filename + ".processed", "w", encoding="utf8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print("  ✓ processed JSON ->", filename + ".processed")


if __name__ == "__main__":
    fetch_all()
    combine_train() 
    print("### Running `process()` …")
    process(str(COMBINED_TRAIN))

    for split in SPLITS:
        test_file = BASE_LOCAL / split / "test_triplets.txt"
        process(str(test_file))



