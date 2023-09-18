import torch.nn as nn

class CitadelNetwork(nn.Module):
    def __init__(self):
        super(CitadelNetwork, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(478, 512)  # Input to Hidden Layer 1
        self.fc2 = nn.Linear(512, 256)  # Hidden Layer 1 to Hidden Layer 2
        self.fc3 = nn.Linear(256, 128)  # Hidden Layer 2 to Hidden Layer 3
        self.fc4 = nn.Linear(128, 1521)  # Hidden Layer 3 to Output
        
        # Define activation function (ReLU for hidden layers)
        self.relu = nn.ReLU()
        
        # Define sigmoid activation for the output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

