# -*- coding: utf-8 -*-

import shutil
import torch

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs,targets)

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_path = best_model_path
        shutil.copyfile(f_path, best_path)