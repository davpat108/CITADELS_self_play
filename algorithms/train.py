import torch
from torch.utils.data import DataLoader, random_split, TensorDataset


def train_masked_model(data, model, epochs=10, learning_rate=0.00, batch_size=64):
    # Check if CUDA is available and select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Training on {device}")
    else:
        print("CUDA is not available. Training on CPU.")
        
    # Transfer model to CUDA device
    model.to(device)
    # Convert data to TensorDataset and DataLoader
    inputs = torch.stack([item[0] for item in data]).to(device)
    labels = torch.stack([item[1] for item in data]).to(device)
    loss_masks = torch.stack([item[2] for item in data]).to(device)
    dataset = TensorDataset(inputs, labels, loss_masks)  # Including the loss_mask in dataset

    # Split dataset into training and evaluation sets
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = torch.nn.MSELoss(reduction='none')  # set reduction='none' to apply loss_mask later
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels, masks in train_loader:  # Including the loss_mask in loader
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            masked_loss = torch.sum(loss * masks) / torch.sum(masks)  # Apply the loss_mask here

            # Backward pass and optimization
            optimizer.zero_grad()
            masked_loss.backward()  # use masked_loss for backpropagation
            optimizer.step()

        # Evaluate the model on the evaluation set
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for inputs, labels, masks in eval_loader:  # Including the loss_mask in loader
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                masked_loss = torch.sum(loss * masks) / torch.sum(masks)  # Apply the loss_mask here
                eval_loss += masked_loss.item()  # use masked_loss for eval_loss calculation

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {masked_loss.item():.4f}, Eval Loss: {eval_loss/len(eval_loader):.4f}")

    print("Training complete.")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

def train_transformer(data, model, epochs, batch_size=64, device='cuda'):
    # Have to figure it how to train with differerent sized inputs and labels while batchsize > 1
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = nn.KLDivLoss(reduction='batchmean')
    log_softmax = nn.LogSoftmax(dim=1)

    # Splitting data into training and evaluation sets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    eval_data = data[train_size:]

    train_batches = split_data_to_batches_by_length(train_data, batch_size)
    eval_batches = split_data_to_batches_by_length(eval_data, batch_size)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        random.shuffle(train_batches)
        for batch in train_batches:
            x_fixed_batch = torch.stack([item[0] for item in batch]).to(device)
            x_variable_batch = torch.stack([item[1] for item in batch]).squeeze(1).to(device)
            labels_fixed_batch = torch.stack([item[2] for item in batch]).to(device)
            labels_variable_batch = torch.stack([item[3] for item in batch]).to(device)

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
        avg_train_loss = total_train_loss / len(train_data)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        # Evaluation phase
        scheduler.step()
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in eval_batches:
                x_fixed_batch = torch.stack([item[0] for item in batch]).to(device)
                x_variable_batch = torch.stack([item[1] for item in batch]).squeeze(1).to(device)
                labels_fixed_batch = torch.stack([item[2] for item in batch]).to(device)
                labels_variable_batch = torch.stack([item[3] for item in batch]).to(device)
                
                outputs_variable, outputs_fixed = model(x_fixed_batch, x_variable_batch)
                outputs_variable_pred = log_softmax(outputs_variable)
                outputs_fixed_pred = log_softmax(outputs_fixed)

                loss_variable = criterion(outputs_variable_pred, labels_variable_batch)
                loss_fixed = criterion(outputs_fixed_pred, labels_fixed_batch)
                total_eval_loss += (loss_variable + loss_fixed).item()

        avg_eval_loss = total_eval_loss / len(eval_data)
        print(f"Epoch {epoch+1}/{epochs} - Eval Loss: {avg_eval_loss:.4f}")


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