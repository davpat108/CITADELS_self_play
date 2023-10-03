import torch
from torch.utils.data import DataLoader, random_split, TensorDataset


def train_masked_model(data, model, epochs=10, learning_rate=0.001, batch_size=64):
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

def train_transformer(model, data, epochs=10, lr=0.001, batch_size=32, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # Assuming you're doing classification

    # Splitting data into training and evaluation sets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    eval_data = data[train_size:]

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)  # Shuffle the training data each epoch

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            x_fixed_batch = torch.stack([item[0] for item in batch])
            x_variable_batch = torch.cat([item[1] for item in batch], dim=1)
            labels_batch = torch.stack([item[2] for item in batch])

            optimizer.zero_grad()
            outputs = model(x_fixed_batch, x_variable_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

        # Evaluation phase
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for i in range(0, len(eval_data), batch_size):
                batch = eval_data[i:i+batch_size]
                x_fixed_batch = torch.stack([item[0] for item in batch])
                x_variable_batch = torch.cat([item[1] for item in batch], dim=1)
                labels_batch = torch.stack([item[2] for item in batch])
                
                outputs = model(x_fixed_batch, x_variable_batch)
                loss = criterion(outputs, labels_batch)
                total_loss += loss.item()

        avg_eval_loss = total_loss / len(eval_data)
        print(f"Epoch {epoch+1}/{epochs} - Eval Loss: {avg_eval_loss:.4f}")

