import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import logging
import torch.nn.functional as F
from algorithms.train_utils import square_and_normalize, log_square_and_normalize
from algorithms.models import ValueOnlyNN
import seaborn as sns
import matplotlib.pyplot as plt
import os
def train_transformer(train_data, val_data, model, epochs, batch_size=64, device='cuda', parent_folder="pretrain", verbose=False):
    # Have to figure it how to train with differerent sized inputs and labels while batchsize > 1
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.KLDivLoss(reduction='batchmean')
    #log_softmax = nn.LogSoftmax(dim=1)


    train_batches = split_data_to_batches_by_length(train_data, batch_size)
    eval_batches = split_data_to_batches_by_length(val_data, batch_size)

    train_losses, eval_losses, learning_rates = [], [], []
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
        
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 0.002:
                param_group['lr'] = 0.002
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
            torch.save(model.state_dict(), parent_folder+"/best_model.pt")
            best_eval_loss = avg_eval_loss
            logging.info("New best model saved")
        if verbose:
            logging.info(f"Epoch {epoch+1}/{epochs} - Eval Loss: {avg_eval_loss:.4f} of which variable: {avg_eval_loss_variable:.4f}, and fixed: {avg_eval_loss_fixed:.4f}")
    
        train_losses.append(avg_train_loss)
        eval_losses.append(avg_eval_loss)
        learning_rates.append(scheduler.get_last_lr()[0])
        
    plot_metrics(train_losses, eval_losses, learning_rates, epochs, parent_folder)
        
        
def train_node_value_only(train_data, val_data, epochs, lr, hidden_size, gamma, batch_size=64, device='cuda', parent_folder="pretrain", verbose=False):
    os.makedirs(parent_folder, exist_ok=True)
    model = ValueOnlyNN(418, hidden_size=hidden_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=gamma)
    criterion = nn.KLDivLoss(reduction='batchmean')

    inputs = torch.stack([item[0] for item in train_data]).to(device)
    labels = torch.stack([item[2] for item in train_data]).to(device)
    training_dataset = TensorDataset(inputs, labels)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    
    inputs = torch.stack([item[0] for item in val_data]).to(device)
    labels = torch.stack([item[2] for item in val_data]).to(device)
    val_dataset = TensorDataset(inputs, labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, eval_losses, learning_rates = [], [], []
    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_inputs, batch_labels in train_dataloader:

            optimizer.zero_grad()
            outputs_fixed = model(batch_inputs)
            #check = square_and_normalize(outputs_fixed)
            labels_pred = square_and_normalize(batch_labels)
            outputs_pred = log_square_and_normalize(outputs_fixed)


            loss = criterion(outputs_pred, labels_pred)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
        if verbose:
            logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        #Evaluation phase
        scheduler.step()
        
        
        

        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_dataloader:
                
                outputs_fixed = model(batch_inputs)
                outputs_pred = log_square_and_normalize(outputs_fixed)
                labels_pred = square_and_normalize(batch_labels)

                loss = criterion(outputs_pred, labels_pred)
                total_eval_loss += loss.item()
        
        avg_eval_loss = total_eval_loss / len(val_dataloader)
        
        if avg_eval_loss < best_eval_loss:
            torch.save(model.state_dict(), parent_folder+"/best_model.pt")
            best_eval_loss = avg_eval_loss
            #logging.info("New best model saved")
        
        if verbose:
            logging.info(f"Epoch {epoch+1}/{epochs} - Eval Loss: {avg_eval_loss:.4f}")
            
        train_losses.append(avg_train_loss)
        eval_losses.append(avg_eval_loss)
        learning_rates.append(scheduler.get_last_lr()[0])
        
    plot_metrics(train_losses, eval_losses, learning_rates, epochs, parent_folder)
    
    return best_eval_loss
                
def plot_metrics(train_losses, eval_losses, learning_rates, epochs, folder):
    plt.figure(figsize=(12, 8))
    
    # Plotting training and evaluation losses
    sns.lineplot(x=range(1, epochs + 1), y=train_losses, label='Train Loss')
    sns.lineplot(x=range(1, epochs + 1), y=eval_losses, label='Eval Loss')

    # Plotting learning rate
    plt.twinx()
    sns.lineplot(x=range(1, epochs + 1), y=learning_rates, label='Learning Rate', color='green')

    plt.title('Training and Evaluation Losses and Learning Rate per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(folder + '/train_metrics.png')


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