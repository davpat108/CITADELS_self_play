import torch.nn as nn
import torch.nn.functional as F

class ValueOnlyNN(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(ValueOnlyNN, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)  
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn2 = nn.BatchNorm1d(int(hidden_size/2))
        self.dropout2 = nn.Dropout(0.2)  
        self.fc3 = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.fc4 = nn.Linear(int(hidden_size/4), 6)      # Output layer without softmax

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x  