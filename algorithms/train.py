import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

def train_model(data, model, epochs=10, learning_rate=0.001, batch_size=64):
    # Convert data to TensorDataset and DataLoader
    inputs = torch.stack([item[0] for item in data])
    labels = torch.stack([item[1] for item in data])
    dataset = TensorDataset(inputs, labels)

    # Split dataset into training and evaluation sets
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()  # Assuming a regression problem; use CrossEntropyLoss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model on the evaluation set
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for inputs, labels in eval_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Eval Loss: {eval_loss/len(eval_loader):.4f}")

    print("Training complete.")
