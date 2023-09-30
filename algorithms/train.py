import torch
from torch.utils.data import DataLoader, random_split, TensorDataset


def train_model(data, model, epochs=10, learning_rate=0.001, batch_size=64):
    # Convert data to TensorDataset and DataLoader
    inputs = torch.stack([item[0] for item in data])
    labels = torch.stack([item[1] for item in data])
    loss_masks = torch.stack([item[2] for item in data])  # Including the loss_mask
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