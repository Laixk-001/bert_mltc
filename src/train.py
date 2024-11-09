# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse

from .model import BERTClass
from .utils import loss_fn, save_ckp
from .load_data import data_loader

val_targets = []
val_outputs = []
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epochs", type=int, default=0)
    parser.add_argument("--n_epoch",type=int, default=3)
    parser.add_argument("--valid_loss_min_input",type=int,default=np.Inf)
    parser.add_argument("--checkpoint_path",type=str,default='./models/current_checkpoint.pt')
    parser.add_argument("--best_model_path",type=str,default="'./models/best_model.pt'")
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--LEARNING_RATE",type=float,default=1e-05)
    return parser.parse_args()


# 模型训练
def train_model(model,optimizer,train_loader,valid_loader,args):
    # train model
    valid_loss_min = args.valid_loss_min

    for epoch in range(args.start_epochs, args.n_epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('Epoch {}: Training start'.format(epoch))

        for batch_idx, data in enumerate(train_loader):
            ids = data['ids'].to(args.device,dtype=torch.long)
            mask = data['mask'].to(args.device,dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(args.device,dtype=torch.long)
            targets = data['targets'].to(args.device,dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, BATCH: {batch_idx}, Training Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        print('Epoch {}: Training End'.format(epoch))
        print('Epoch {}: Validation Start'.format(epoch))
    
        # eval model
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader, 0):
                ids = data['ids'].to(args.device, dtype=torch.long)
                mask = data['mask'].to(args.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(args.device, dtype=torch.long)
                targets = data['targets'].to(args.device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            print('Epoch {}: Validation End'.format(epoch))
            train_loss = train_loss / len(train_loader)
            valid_loss = valid_loss / len(valid_loader)
            # print training/validation statistics
            print('Epoch: {} \t Avgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'
                    .format(epoch, train_loss, valid_loss))
        
            # create checkpoint
            check_point = {
                'epoch':epoch+1,
                'valid_loss_min': valid_loss_min,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save checkpoint
            save_ckp(check_point,False,args.checkpoint_path,args.best_model_path)
            # save model
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased from {:.6f} to {:.6f}). Saving model'
                      .format(valid_loss_min, valid_loss))
                # save checkpoint as best model
                save_ckp(check_point, True, args.checkpoint_path, args.best_model_path)
                valid_loss_min = valid_loss
        print('Epoch {}  Done\n'.format(epoch))
    return model

if __name__ == '__main__':
    args = parse_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    # 加载数据集
    training_loader, validation_loader = data_loader('./data/train.csv')
    # 创建模型
    model = BERTClass()
    model.to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.LEARNING_RATE)
    # 模型训练
    trained_model = train_model(model,optimizer,training_loader,validation_loader,args)
    # 模型预测指标
    val_predicts = (np.array(val_outputs) >= 0.5).astype(int)
    accuracy = accuracy_score(val_targets, val_predicts)
    f1_score_micro = f1_score(val_targets, val_predicts, average='micro')
    f1_score_macro = f1_score(val_targets, val_predicts, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(classification_report(val_targets, val_predicts))