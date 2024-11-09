# -*- coding: utf-8 -*-

import torch
import transformers

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL_NAME_OR_PATH",type=str,default="./bert-base-uncased")
    parser.add_argument("--HIDDEN_LAYER_SIZE",type=int,default=500)
    return parser.parse_args()

# 只修改最后一层输出
class BERTClass(torch.nn.Module):
    def __init__(self,model_path,hidden_layer_size):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model_path)
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(hidden_layer_size, 6)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_l1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_l2 = self.l2(output_l1)
        output = self.l3(output_l2)
        return output
    