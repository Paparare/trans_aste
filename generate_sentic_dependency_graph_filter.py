from pathlib import Path
import requests
import sys
import json
import pickle
import numpy as np
import spacy
from typing import Set

from dataloader import read_data

BASE_LOCAL = Path("SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020")
REPO_RAW = "https://raw.githubusercontent.com/xuuuluuu/Position-Aware-Tagging-for-ASTE/master/data/ASTE-Data-V2"
SPLITS = ["14lap", "14res", "15res", "16res"]
FILES = ["train_triplets.txt", "dev_triplets.txt", "test_triplets.txt"]
COMBINED_TRAIN = BASE_LOCAL / "combined_train.txt"
DEDUP_SUFFIX = "_dedup.txt"
CHUNK = 16 * 1024
nlp = spacy.load("en_core_web_lg")


def download_file(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=15) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)


def fetch_all() -> None:
    for split in SPLITS:
        for fname in FILES:
            url = f"{REPO_RAW}/{split}/{fname}"
            download_file(url, BASE_LOCAL / split / fname)


def combine_train() -> None:
    with COMBINED_TRAIN.open("w", encoding="utf8") as fout:
        for split in SPLITS:
            src = BASE_LOCAL / split / "train_triplets.txt"
            lines = src.read_text(encoding="utf8").splitlines()
            fout.write("\n".join(lines) + "\n")


def load_sentence_set(txt_path: Path) -> Set[str]:
    sents = set()
    with txt_path.open(encoding="utf8") as f:
        for line in f:
            if "####" not in line:
                continue
            sentence, _ = line.split("####", 1)
            sents.add(sentence.strip())
    return sents


def dedup_test_file(test_path: Path, train_sents: Set[str]) -> Path:
    out_path = test_path.with_name(test_path.stem + DEDUP_SUFFIX)
    with test_path.open("r", encoding="utf8") as fin, out_path.open("w", encoding="utf8") as fout:
        for line in fin:
            if "####" not in line:
                continue
            sentence, _ = line.split("####", 1)
            if sentence.strip() in train_sents:
                continue
            fout.write(line)
    return out_path


def load_sentic_word():
    path = "./senticNet/senticnet_word.txt"
    senticNet = {}
    with open(path, "r", encoding="utf8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            word, sentic = line.split("\t")
            senticNet[word] = sentic
    return senticNet


def dependency_adj_matrix(text: str, senticNet):
    text = text.lower()
    doc = nlp(text)
    L = len(doc)
    mat = np.zeros((L, L), dtype="float32")
    tokens = []
    for token in doc:
        tokens.append(str(token))
        sentic = float(senticNet.get(str(token), 0)) + 1
        mat[token.i][token.i] = sentic
        for child in token.children:
            mat[token.i][child.i] = sentic
            mat[child.i][token.i] = sentic
    return mat, tokens


def process(filename: str):
    senticNet = load_sentic_word()
    sentences, truth_pairs = read_data(filename)
    idx2graph = {}
    sent_tokens = []
    for idx, sent in enumerate(sentences):
        graph, toks = dependency_adj_matrix(sent, senticNet)
        idx2graph[idx] = graph
        sent_tokens.append(toks)
    with open(filename + ".graph", "wb") as f:
        pickle.dump(idx2graph, f)
    processed = []
    for toks, pairs in zip(sent_tokens, truth_pairs):
        processed.append({"sentence": " ".join(toks), "truth_pairs": list(set(pairs))})
    with open(filename + ".processed", "w", encoding="utf8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    fetch_all()
    combine_train()
    train_16 = BASE_LOCAL / "16res" / "train_triplets.txt"
    test_15 = BASE_LOCAL / "15res" / "test_triplets.txt"
    train16_sentences = load_sentence_set(train_16)
    test_15_dedup = dedup_test_file(test_15, train16_sentences)
    process(str(COMBINED_TRAIN))
    process(str(BASE_LOCAL / "14lap" / "test_triplets.txt"))
    process(str(BASE_LOCAL / "14res" / "test_triplets.txt"))
    process(str(test_15_dedup))
    process(str(BASE_LOCAL / "16res" / "test_triplets.txt"))
