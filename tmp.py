import argparse

config = argparse.ArgumentParser()
config.add_argument("--gcn-add-bert-output", action='store_true')

config = config.parse_args()

print(config.gcn_add_bert_output)