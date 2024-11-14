#!/bin/bash
{

CUDA_VISIBLE_DEVICES=3 python train.py \
    --train_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt.processed \
    --dev_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/dev_triplets.txt.processed \
    --test_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt.processed


CUDA_VISIBLE_DEVICES=3 python train.py \
    --train_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/train_triplets.txt.processed \
    --dev_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/dev_triplets.txt.processed \
    --test_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/test_triplets.txt.processed 

CUDA_VISIBLE_DEVICES=3 python train.py \
    --train_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/train_triplets.txt.processed \
    --dev_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/dev_triplets.txt.processed \
    --test_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/test_triplets.txt.processed 

CUDA_VISIBLE_DEVICES=3 python train.py \
    --train_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/train_triplets.txt.processed \
    --dev_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/dev_triplets.txt.processed \
    --test_dataset_path SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/test_triplets.txt.processed 

exit
}
