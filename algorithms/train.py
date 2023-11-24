import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import logging
import torch.nn.functional as F
from algorithms.train_utils import square_and_normalize, log_square_and_normalize

def train_transformer(train_data, val_data, model, epochs, batch_size=64, device='cuda', best_model_name="best_model.pt", verbose=False):
    # Have to figure it how to train with differerent sized inputs and labels while batchsize > 1
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.KLDivLoss(reduction='batchmean')
    #log_softmax = nn.LogSoftmax(dim=1)


    train_batches = split_data_to_batches_by_length(train_data, batch_size)
    eval_batches = split_data_to_batches_by_length(val_data, batch_size)
    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_loss_variable = 0
        total_train_loss_fixed = 0
        random.shuffle(train_batches)
        for batch in train_batches:
            x_fixed_batch = torch.stack([item[0] for item in batch]).to(device)
            x_variable_batch = torch.stack([item[1] for item in batch]).squeeze(1).to(device)
            labels_fixed_batch = torch.stack([square_and_normalize(item[2], dim=-1) for item in batch]).to(device)
            labels_variable_batch = torch.stack([square_and_normalize(item[3], dim=-1) for item in batch]).to(device)

            optimizer.zero_grad()
            outputs_variable, outputs_fixed = model(x_fixed_batch, x_variable_batch)
            check = square_and_normalize(outputs_fixed)
            outputs_variable_pred = log_square_and_normalize(outputs_variable)
            outputs_fixed_pred = log_square_and_normalize(outputs_fixed)


            loss_variable = criterion(outputs_variable_pred, labels_variable_batch)
            loss_fixed = criterion(outputs_fixed_pred, labels_fixed_batch)
            total_loss = loss_variable + loss_fixed
            
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()
            total_train_loss_variable += loss_variable.item()
            total_train_loss_fixed += loss_fixed.item()
            
            
        avg_train_loss = total_train_loss / len(train_batches)
        avg_train_loss_variable = total_train_loss_variable / len(train_batches)
        avg_train_loss_fixed = total_train_loss_fixed / len(train_batches)
        
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
        if verbose:
            logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, of which variable: {avg_train_loss_variable:.4f}, and fixed: {avg_train_loss_fixed:.4f}")
        
        #Evaluation phase
        scheduler.step()
        model.eval()
        total_eval_loss = 0
        total_eval_loss_variable = 0
        total_eval_loss_fixed = 0
        with torch.no_grad():
            for batch in eval_batches:
                x_fixed_batch = torch.stack([item[0] for item in batch]).to(device)
                x_variable_batch = torch.stack([item[1] for item in batch]).squeeze(1).to(device)
                labels_fixed_batch = torch.stack([square_and_normalize(item[2], dim=-1) for item in batch]).to(device)
                labels_variable_batch = torch.stack([square_and_normalize(item[3], dim=-1) for item in batch]).to(device)
                
                outputs_variable, outputs_fixed = model(x_fixed_batch, x_variable_batch)
                outputs_variable_pred = log_square_and_normalize(outputs_variable)
                outputs_fixed_pred = log_square_and_normalize(outputs_fixed)

                loss_variable = criterion(outputs_variable_pred, labels_variable_batch)
                loss_fixed = criterion(outputs_fixed_pred, labels_fixed_batch)
                total_eval_loss += (loss_variable + loss_fixed).item()
                total_eval_loss_variable += loss_variable.item()
                total_eval_loss_fixed += loss_fixed.item()

        avg_eval_loss = total_eval_loss / len(eval_batches)
        avg_eval_loss_variable = total_eval_loss_variable / len(eval_batches)
        avg_eval_loss_fixed = total_eval_loss_fixed / len(eval_batches)

        if avg_eval_loss < best_eval_loss:
            torch.save(model.state_dict(), best_model_name)
            best_eval_loss = avg_eval_loss
            logging.info("New best model saved")
        if verbose:
            pass
            #logging.info(f"Epoch {epoch+1}/{epochs} - Eval Loss: {avg_eval_loss:.4f} of which variable: {avg_eval_loss_variable:.4f}, and fixed: {avg_eval_loss_fixed:.4f}")

    return [best_eval_loss], [best_train_loss]


def split_data_to_batches_by_length(data, batch_size):
    # Group data by the length of x_variable
    groups = {}
    for item in data:
        key = item[1].shape[1]
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    # Split grouped data into batches and collect them in a list
    all_batches = []
    for key in groups:
        group = groups[key]
        random.shuffle(group)
        
        for i in range(0, len(group), batch_size):
            batch = group[i:i+batch_size]
            all_batches.append(batch)

    return all_batches