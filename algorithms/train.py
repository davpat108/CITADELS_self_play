import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import logging
import torch.nn.functional as F


def train_transformer(data, model, epochs, batch_size=64, device='cuda', train_index=0):
    # Have to figure it how to train with differerent sized inputs and labels while batchsize > 1
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.KLDivLoss(reduction='batchmean')
    log_softmax = nn.LogSoftmax(dim=1)

    # Splitting data into training and evaluation sets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    eval_data = data[train_size:]

    train_batches = split_data_to_batches_by_length(train_data, batch_size)
    eval_batches = split_data_to_batches_by_length(eval_data, batch_size)
    best_eval_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        random.shuffle(train_batches)
        for batch in train_batches:
            x_fixed_batch = torch.stack([item[0] for item in batch]).to(device)
            x_variable_batch = torch.stack([item[1] for item in batch]).squeeze(1).to(device)
            labels_fixed_batch = torch.stack([F.softmax(item[2], dim=-1) for item in batch]).to(device)
            labels_variable_batch = torch.stack([F.softmax(item[3], dim=-1) for item in batch]).to(device)

            optimizer.zero_grad()
            outputs_variable, outputs_fixed = model(x_fixed_batch, x_variable_batch)
            outputs_variable_pred = log_softmax(outputs_variable)
            outputs_fixed_pred = log_softmax(outputs_fixed)


            loss_variable = criterion(outputs_variable_pred, labels_variable_batch)
            loss_fixed = criterion(outputs_fixed_pred, labels_fixed_batch)
            total_loss = loss_variable + loss_fixed
            
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()
            
        avg_train_loss = total_train_loss / len(train_batches)
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        # Evaluation phase
        scheduler.step()
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in eval_batches:
                x_fixed_batch = torch.stack([item[0] for item in batch]).to(device)
                x_variable_batch = torch.stack([item[1] for item in batch]).squeeze(1).to(device)
                labels_fixed_batch = torch.stack([F.softmax(item[2], dim=-1) for item in batch]).to(device)
                labels_variable_batch = torch.stack([F.softmax(item[3], dim=-1) for item in batch]).to(device)
                
                outputs_variable, outputs_fixed = model(x_fixed_batch, x_variable_batch)
                outputs_variable_pred = log_softmax(outputs_variable)
                outputs_fixed_pred = log_softmax(outputs_fixed)

                loss_variable = criterion(outputs_variable_pred, labels_variable_batch)
                loss_fixed = criterion(outputs_fixed_pred, labels_fixed_batch)
                total_eval_loss += (loss_variable + loss_fixed).item()

        avg_eval_loss = total_eval_loss / len(eval_batches)

        if avg_eval_loss < best_eval_loss:
            torch.save(model.state_dict(), f"best_model{train_index}.pt")
            best_eval_loss = avg_eval_loss
            logging.info("New best model saved")

        logging.info(f"Epoch {epoch+1}/{epochs} - Eval Loss: {avg_eval_loss:.4f}")

    return [best_eval_loss]


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