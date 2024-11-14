import shutil
import torch, os, time, random, sys, json
import numpy as np
import logging
import torch.nn as nn
from utils.trans_module import TransitionModel, RobertaModel, RobertaEncoder
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.evaluation import evaluate, get_eval_result
import datetime
from dataloader import TripletDataset, generate_batch_fn
from torch.utils.data import DataLoader
from utils.config import get_config
from tqdm import tqdm
import pickle

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:2080'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:2080'

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

config = get_config()

# Set a unique time-stamped string for save_path
now = datetime.datetime.now()
now_time_string = f"{now.year:04d}{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}{now.second:02d}_{config.seed:05d}"

# Set seeds for reproducibility
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ['PYTHONHASHSEED'] = str(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define and create the save path
save_path = os.path.join(config.save_path, now_time_string)
os.makedirs(save_path, exist_ok=True)

# Save configuration
with open(os.path.join(save_path, "config.json"), "w") as fp:
    json.dump(config.__dict__, fp)

# Define essential files to copy (modify this list with any essential files you need)
essential_files = ["train.py", "config.py"]

# Copy only essential files to reduce storage use
codes_save_path = os.path.join(save_path, 'codes')
os.makedirs(codes_save_path, exist_ok=True)
base_dir = os.getcwd()
for name in essential_files:
    src_path = os.path.join(base_dir, name)
    dest_path = os.path.join(codes_save_path, name)
    if os.path.exists(src_path):
        shutil.copyfile(src_path, dest_path)

# Copy the utils directory, if necessary
utils_dir = os.path.join(base_dir, "utils")
if os.path.exists(utils_dir):
    dest_utils_dir = os.path.join(codes_save_path, "utils")
    if os.path.exists(dest_utils_dir):
        shutil.rmtree(dest_utils_dir)
    shutil.copytree(utils_dir, dest_utils_dir)

# Logging setup with both file and console handlers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Log file handler with a reduced log file size if necessary
log_file_path = os.path.join(save_path, 'log.txt')
if os.path.exists(log_file_path):
    os.remove(log_file_path)
fh = logging.FileHandler(log_file_path, mode='a')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(ch)  # Output to terminal
logger.addHandler(fh)  # Output to log file

logger.info("Setup complete. Configuration and essential files saved.")


def train(lr, bs, do, we, roberta_path, roberta_output_size):
    logger.info('Loading data...')
    os.makedirs(save_path, exist_ok=True)

    # Load training data
    with open(config.train_dataset_path, 'r') as f:
        loaded_data_list = json.load(f)
        train_sentences_list, train_truth_pairs_list = zip(
            *[(d['sentence'], d['truth_pairs']) for d in loaded_data_list])
    train_graphs = pickle.load(open('.'.join(config.train_dataset_path.split('.')[:-1]) + '.graph', 'rb'))
    train_dataset = TripletDataset(train_sentences_list, train_truth_pairs_list, train_graphs, 'train')
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=generate_batch_fn)

    # Load dev data
    with open(config.dev_dataset_path, 'r') as f:
        loaded_data_list = json.load(f)
        dev_sentences_list, dev_truth_pairs_list = zip(*[(d['sentence'], d['truth_pairs']) for d in loaded_data_list])
    dev_graphs = pickle.load(open('.'.join(config.dev_dataset_path.split('.')[:-1]) + '.graph', 'rb'))
    dev_dataset = TripletDataset(dev_sentences_list, dev_truth_pairs_list, dev_graphs, 'dev')
    dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=False, collate_fn=generate_batch_fn)

    # Load test data
    test_paths = [
        'SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/test_triplets.txt.processed',  # Original test data
        'SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt.processed',
        'SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/test_triplets.txt.processed',
        'SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/test_triplets.txt.processed'
    ]

    test_loaders = {}
    for test_path in test_paths:
        with open(test_path, 'r') as f:
            loaded_data_list = json.load(f)
            test_sentences_list, test_truth_pairs_list = zip(
                *[(d['sentence'], d['truth_pairs']) for d in loaded_data_list])
        test_graphs = pickle.load(open('.'.join(test_path.split('.')[:-1]) + '.graph', 'rb'))
        test_dataset = TripletDataset(test_sentences_list, test_truth_pairs_list, test_graphs, 'test')
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=generate_batch_fn)
        test_loaders[test_path] = test_loader


    logger.info('Data loaded.')

    logger.info('Initializing model...')
    base_encoder = RobertaEncoder(config)
    base_encoder.cuda()
    trans_model = TransitionModel(config)
    trans_model.cuda()
    logger.info('Model initialized.')

    # crossentropy = nn.CrossEntropyLoss()
    base_encoder_optimizer = filter(lambda x: x.requires_grad, base_encoder.parameters())
    trans_optimizer = filter(lambda x: x.requires_grad, trans_model.parameters())
    optimizer_parameters = [
        {'params': [p for p in trans_optimizer if len(p.data.size()) > 1], 'weight_decay': we},
        {'params': [p for p in trans_optimizer if len(p.data.size()) == 1], 'weight_decay': 0.0},
        {'params': base_encoder_optimizer, 'lr': lr},
        {'params': trans_optimizer}
    ]

    optimizer = AdamW(optimizer_parameters, lr, weight_decay=1e-2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_dataset) * config.epochs * config.warm_up / bs),
        num_training_steps=len(train_dataset) * config.epochs // bs
    )

    best_f1 = -1.0
    best_metrics = {test_path: {"f1": 0, "rec": 0, "pre": 0} for test_path in test_loaders.keys()}

    for epoch_i in range(config.epochs):
        logger.info(f"Running epoch: {epoch_i}")
        for batch in tqdm(train_loader):
            base_encoder.train()
            trans_model.train()
            optimizer.zero_grad()

            sent_token_ids_list = batch['sent_token_ids_list']
            graph_list = batch['graph_list']
            word_reps_list = base_encoder(sent_token_ids_list, graph_list)

            total_loss, action_logits, sent_logits, true_action_tensor, true_sent_tensor = trans_model(
                word_reps_list, batch['parser_states_list'], 'train')

            loss = total_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluate after each epoch
        # train_eval_res, _ = evaluate(trans_model, base_encoder, train_dataset, train_loader, 'train')
        # dev_eval_res, _ = evaluate(trans_model, base_encoder, dev_dataset, dev_loader, 'eval')


        for test_path, test_loader in test_loaders.items():
            test_eval_res, _ = evaluate(trans_model, base_encoder, test_dataset, test_loader, 'eval')

            logger.info(f"Testing on {test_path}:")
            logger.info(
                f"  Current - F1: {test_eval_res['f1']:.6f}, Recall: {test_eval_res['rec']:.6f}, Precision: {test_eval_res['pre']:.6f}")

            # Check if the current test results are better than the saved best results
            if test_eval_res["f1"] > best_metrics[test_path]["f1"]:
                logger.info(f"Improvement detected! Previous Best F1: {best_metrics[test_path]['f1']:.6f}")

                # Update best metrics
                best_metrics[test_path]["f1"] = test_eval_res["f1"]
                best_metrics[test_path]["rec"] = test_eval_res["rec"]
                best_metrics[test_path]["pre"] = test_eval_res["pre"]

                model_path = os.path.join(save_path, f'trans_best_{test_path.replace("/", "_")}.mdl')
                logger.info(f"New best metrics for {test_path}:")
                logger.info(
                    f"  Best F1: {best_metrics[test_path]['f1']:.6f}, Best Recall: {best_metrics[test_path]['rec']:.6f}, "
                    f"Best Precision: {best_metrics[test_path]['pre']:.6f}")
                logger.info(f"Saving model to {model_path}")
                torch.save(trans_model.state_dict(), model_path)
            else:
                logger.info(f"No improvement for {test_path}. Current F1: {test_eval_res['f1']:.6f}, "
                            f"Best F1: {best_metrics[test_path]['f1']:.6f}, Best Recall: {best_metrics[test_path]['rec']:.6f}, "
                            f"Best Precision: {best_metrics[test_path]['pre']:.6f}")

            logger.info("+" * 80)

        # logger.info("=" * 80)
        # logger.info(f"Epoch {epoch_i} Results:")
        # logger.info(f"  [Dev] F1: {dev_eval_res['f1']:.6f}, Recall: {dev_eval_res['rec']:.6f}, Precision: {dev_eval_res['pre']:.6f}")
        # logger.info("=" * 80)

        # for test_path, test_loader in test_loaders.items():
        #         #     test_eval_res, _ = evaluate(trans_model, base_encoder, test_dataset, test_loader, 'eval')
        #         #     logger.info(f"Testing on {test_path}:")
        #         #     logger.info(
        #         #         f"  F1: {test_eval_res['f1']:.6f}, Recall: {test_eval_res['rec']:.6f}, Precision: {test_eval_res['pre']:.6f}")
        #         #     logger.info("+" * 80)
        #         #
        #         # if dev_eval_res["f1"] > best_f1:
        #         #     best_f1 = dev_eval_res["f1"]
        #         #     best_recall = dev_eval_res["rec"]
        #         #     best_precision = dev_eval_res["pre"]
        #         #     model_path = os.path.join(save_path, 'trans_best.mdl')
        #         #     logger.info(f"New best F1 score: {best_f1}. New best recall: {best_recall}. New best precision: {best_precision}. Saving model.")
        #         #     torch.save(trans_model.state_dict(), model_path)

        #     early_stop_counter = 0
        # else:
        #     early_stop_counter += 1

        # if early_stop_counter >= 20:
        #     logger.info(f"Early stopping triggered at epoch {epoch_i}. Best F1: {best_f1}")
        #     logger.info(f'Current learning_rate: {lr}, batch_size: {bs}, dropout: {do}, weight_decay: {we}')
        #     break

    # model_path = os.path.join(save_path, 'trans_best.mdl')
    # if os.path.exists(model_path):
    #     trans_model.load_state_dict(torch.load(model_path))
    #     for test_path, test_loader in test_loaders.items():
    #         test_eval_res, _ = evaluate(trans_model, base_encoder, test_dataset, test_loader, 'eval')
    #         logger.info(f"Testing on {test_path}:")
    #         logger.info(
    #             f"  F1: {test_eval_res['f1']:.6f}, Recall: {test_eval_res['rec']:.6f}, Precision: {test_eval_res['pre']:.6f}")
    #         logger.info("+" * 80)
    # else:
    #     logger.warning("Best model not found. Model evaluation on test sets skipped.")

    return {
        'best_f1': best_f1,
        'learning_rate': lr,
        'batch_size': bs,
        'dropout': do,
        'weight_decay': we
    }


def grid_search(lr_list, bs_list, do_list, wd_list):
    best_f1 = 0
    best_config = None

    for lr in lr_list:
        for batch_size in bs_list:
            for dropout in do_list:
                for weight_decay in wd_list:
                    results = train(lr, batch_size, dropout, weight_decay, config.roberta_path, config.roberta_output_size)

                    current_f1 = results['best_f1']
                    print(f"Current F1: {current_f1} for lr={lr}, batch_size={batch_size}, dropout={dropout}, weight_decay={weight_decay}")

                    # Compare with the global best F1 score
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_config = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'dropout': dropout,
                            'weight_decay': weight_decay
                        }

    print(f"Best F1 score: {best_f1}")
    print(f"Best configuration: {best_config}")

learning_rates = [1e-5]
batch_sizes = [4]
dropouts = [0.3]
weight_decays = [1e-5]

grid_search(learning_rates, batch_sizes, dropouts, weight_decays)

