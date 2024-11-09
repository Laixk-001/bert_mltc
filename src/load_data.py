# -*- coding: utf-8 -*-

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from typing import Dict
import argparse

# 配置设置
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LABELS", type=Dict[str], default=['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'],help="")
    parser.add_argument("--MODEL_NAME_OR_PATH",type=str,default="./bert-base-uncased")
    parser.add_argument("--MAX_LEN",type=int,default=128)
    parser.add_argument("--TRAIN_BATCH_SIZE",type=int,default=32)
    parser.add_argument("--VALID_BATCH_SIZE",type=int,default=32)
    return parser.parse_args()



# Torch 数据集导入
class CustomDataset:
    def __init__(self,dataframe,tokenizer,max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.title = dataframe['TITLE']
        self.abstract = dataframe['ABSTRACT']
        self.target = self.data['target_list']
    
    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        abstract = str(self.abstract[index])

        inputs = self.tokenizer.encode_plus(
            title,
            abstract,
            add_special_tokens = True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation='only_second'
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
               }

def data_loader(raw_csv_path,params):
    df_raw = pd.read_csv(raw_csv_path)
    df_raw['target_list'] = df_raw[params.LABELS].values.to_list()
    df = df_raw[['TITLE'],['ABSTRACT'],['target_list']].copy()
    
    train_size = 0.8
    train_dataset = df.sample(frac=train_dataset, random_state=200)
    valid_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    print(f"FULL Dataset: {df.shape}, "
          f"TRAIN Dataset: {train_dataset.shape}, "
          f"TEST Dataset: {valid_dataset.shape}")
    tokenizer = BertTokenizer.from_pretrained(params.MODEL_NAME_OR_PATH)
    train_set = CustomDataset(train_dataset,tokenizer,params.MAX_LEN)
    valid_set = CustomDataset(valid_dataset,tokenizer,params.MAX_LEN)
    train_parmas = {'batch_size':params.TRAIN_BATCH_SIZE,'shuffle':True,'num_workers':0}
    valid_parmas = {'batch_size':params.VALID_BATCH_SIZE,'shuffle':False,'num_workers':0}    
    train_loader = DataLoader(train_set, **train_parmas)
    valid_loader = DataLoader(valid_set,**valid_parmas)
    return train_loader,valid_loader

if __name__ == '__main__':
    args = parse_args()
    raw_csv_path = "./data/train.csv"
    training_loader, validation_loader = data_loader('./data/train.csv',args)
    