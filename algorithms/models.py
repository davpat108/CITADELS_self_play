import torch
import torch.nn as nn
import torch.nn.init as init

class CitadelNetwork(nn.Module):
    def __init__(self):
        super(CitadelNetwork, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(478, 512)  # Input to Hidden Layer 1
        self.bn1 = nn.BatchNorm1d(512)  # BatchNorm for Hidden Layer 1
        self.dropout1 = nn.Dropout(0.3)  # Dropout for Hidden Layer 1
        
        self.fc2 = nn.Linear(512, 1024)  # Hidden Layer 1 to Hidden Layer 2
        self.bn2 = nn.BatchNorm1d(1024)  # BatchNorm for Hidden Layer 2
        self.dropout2 = nn.Dropout(0.3)  # Dropout for Hidden Layer 2
        
        self.fc3 = nn.Linear(1024, 1024)  # Hidden Layer 2 to Hidden Layer 3
        self.bn3 = nn.BatchNorm1d(1024)  # BatchNorm for Hidden Layer 3
        self.dropout3 = nn.Dropout(0.3)  # Dropout for Hidden Layer 3

        # Apply Kaiming Initialization to Linear Layers with ReLU Activation
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        

        self.fc4 = nn.Linear(1024, 1521)  # Hidden Layer 3 to Output
        init.xavier_normal_(self.fc4.weight)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x
    
    def relu(self, x):
        return nn.functional.relu(x, inplace=True)

    def sigmoid(self, x):
        return torch.sigmoid(x)
    

import torch.nn.functional as F
class VariableInputNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size):
        super(VariableInputNN, self).__init__()
        
        self.embedding = nn.Linear(input_size, hidden_size)  # Assumes input is not categorical
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x_fixed, x_variable):

        mask = (x_variable.sum(dim=-1) == 0)
        x_variable = self.embedding(x_variable)
        x_variable = self.transformer_encoder(x_variable, src_key_padding_mask=mask)
        x_variable_aggregated = x_variable.mean(dim=1)
        
        x_combined = torch.cat([x_fixed, x_variable_aggregated], dim=-1)
    
        x = F.relu(self.fc1(x_combined))
        output = F.softmax(self.fc2(x), dim=-1)
        
        return output
