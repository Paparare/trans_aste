import torch, os, time, random, sys, json
import numpy as np
import logging
import torch.nn as nn
sys.path.append('./utils')
from trans_module import TransitionModel, BertEncoder
from transformers import AdamW, get_linear_schedule_with_warmup
from evaluation import evaluate, get_eval_result
import datetime
from dataloader import TripletDataset, generate_batch_fn, read_data
from torch.utils.data import DataLoader
import pandas as pd
from config import get_config
from tqdm import tqdm
import json
config = get_config()

now = datetime.datetime.now()
now_time_string = "{:0>4d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}_{:0>5d}".format(
                now.year, now.month, now.day, now.hour, now.minute, now.second, config.seed)

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ['PYTHONHASHSEED'] = str(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_sentences_list, train_truth_pairs_list = read_data(config.train_dataset_path)
train_dataset = TripletDataset(train_sentences_list, train_truth_pairs_list, 'train')
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=generate_batch_fn)

dev_sentences_list, dev_truth_pairs_list = read_data(config.dev_dataset_path)
dev_dataset = TripletDataset(dev_sentences_list, dev_truth_pairs_list, 'dev')
dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=generate_batch_fn)

test_sentences_list, test_truth_pairs_list = read_data(config.test_dataset_path)
test_dataset = TripletDataset(test_sentences_list, test_truth_pairs_list, 'test')
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=generate_batch_fn)

train_len = len(train_dataset)
train_iter_len = (train_len // config.batch_size) + 1
if train_len % config.batch_size == 1:
    train_iter_len -= 1
num_training_steps = train_iter_len * config.epochs
num_warmup_steps = int(num_training_steps * config.warm_up)
print('Data loaded.')

print('Initializing model...')
base_encoder = BertEncoder(config)
base_encoder.cuda()
trans_model = TransitionModel(config)
trans_model.cuda()
print('Model initialized.')

save_path = '/home/baojianzhu/workspace/trans_aste/saved_models/20230813_145523_00000'

# test set results
base_encoder.load_state_dict(torch.load(os.path.join(save_path, 'bert_best.mdl')))
trans_model.load_state_dict(torch.load(os.path.join(save_path, 'trans_best.mdl')))
print('='*20 +'The performance on test set' + '='*20)
test_eval_res, all_pred_pairs = evaluate(trans_model, base_encoder, test_dataset, test_loader, 'eval')
res_msg = get_eval_result(test_eval_res)
print(test_eval_res)

rel_f1 = test_eval_res["f1"]
with open(os.path.join(save_path, 'result.txt'), 'w') as fp:
    fp.write(res_msg + '\n')

with open(os.path.join(save_path, 'pred_test_res.json'), 'w') as fp:
    json.dump(all_pred_pairs, fp)
# test_res_df.to_csv(os.path.join(save_path, "test_pred_res.csv"), index=False)