import argparse


def get_config():
    config = argparse.ArgumentParser()
    config.add_argument("--dataset-name", default="CDCP", type=str)
    config.add_argument("--train_dataset_path",
                        default="SemEval-Triplet-data/Integrated-Splits/train_triplets.txt.processed",
                        type=str)
    config.add_argument("--dev_dataset_path",
                        default="SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/dev_triplets.txt.processed",
                        type=str)
    config.add_argument("--test_dataset_path",
                        default="SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/test_triplets.txt.processed",
                        type=str)
    config.add_argument("--save-path", default="./saved_models", type=str)

    # model paths
    config.add_argument("--roberta-path", default="roberta-base", type=str,
                        help="Path or name of the pretrained RoBERTa model")

    # training
    config.add_argument("--use_layer_norm", default=True, type=bool, help="Whether to use Layer Normalization")
    config.add_argument("--layer_norm_eps", default=1e-5, type=float, help="Epsilon value for Layer Normalization")
    config.add_argument("--device", default='1', type=str)
    config.add_argument("--seed", default=111, type=int)
    config.add_argument("--epochs", default=2000, type=int)
    config.add_argument("--showtime", default=50, type=int)
    config.add_argument("--finetune_lr", default=1e-3, type=float)
    config.add_argument("--warm-up", default=5e-2, type=float)
    config.add_argument("--early-num", default=3, type=int)
    config.add_argument("--dropout", default=0.3, type=float)

    # trans model param
    config.add_argument("--cell-size", default=256, type=int)
    config.add_argument("--lstm-layers", default=1, type=int)
    config.add_argument("--is-bi", default=True, type=bool)
    config.add_argument("--roberta-output-size", default=768, type=int,
                        help="Output size of RoBERTa model (hidden size)")
    config.add_argument("--mlp-size", default=512, type=int)
    config.add_argument("--scale-factor", default=2, type=int)
    config.add_argument("--max-AC-num", default=12, type=int)
    config.add_argument("--position-ebd-dim", default=256, type=int)
    config.add_argument("--action-ebd-dim", default=256, type=int)
    config.add_argument("--action-type-num", default=8, type=int)
    config.add_argument("--action-label-num", default=6, type=int)
    config.add_argument("--AC-type-label-num", default=4, type=int)
    config.add_argument("--position-trainable", default=True, type=bool)
    config.add_argument("--action-trainable", default=True, type=bool)
    config.add_argument("--max-dist-len", default=100, type=int)
    config.add_argument("--max-grad-norm", default=1.0, type=float)

    config.add_argument("--gcn-layer-num", default=0, type=int)
    config.add_argument("--gcn-add-roberta-output", action='store_true', help="Add RoBERTa output to the GCN output")
    config.add_argument("--attn", action='store_true')

    config = config.parse_args()

    if config.dataset_name == 'PE':
        config.AC_type2id = {"MajorClaim": 0, "Claim": 1, "Premise": 2}
    elif config.dataset_name == 'CDCP':
        config.AC_type2id = {"value": 0, "testimony": 1, "policy": 2, "fact": 3, "reference": 4}
    config.para_type2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}

    return config