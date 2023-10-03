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
    

import torch.nn as nn
from transformers import Transformer

class VariableInputNN(nn.Module):
    def __init__(self, option_encoding_size, variable_item_size, transformer_nhead=2, num_transformer_layers=2):
        super(VariableInputNN, self).__init__()
        
        self.embedding_fixed = nn.Linear(option_encoding_size, variable_item_size)
        self.transformer = Transformer(d_model=variable_item_size, nhead=transformer_nhead, num_encoder_layers=num_transformer_layers)
        self.aggregate = nn.AdaptiveAvgPool1d(1)  # Using average pooling for aggregation
        self.fc_out = nn.Linear(variable_item_size, 1)  # This produces a scalar output, to be expanded later.

    def forward(self, x_fixed, x_variable):
        x_fixed = self.embedding_fixed(x_fixed).unsqueeze(0)  # shape: [1, batch_size, variable_item_size]
        
        # Concatenate the fixed vector with variable vectors
        combined_seq = torch.cat([x_fixed, x_variable], dim=0)
        
        # Pass through the transformer
        transformer_output = self.transformer(combined_seq, combined_seq)
        
        # Aggregate across sequence dimension
        aggregated_representation = self.aggregate(transformer_output.permute(1, 2, 0)).squeeze(-1)  # shape: [batch_size, variable_item_size]

        # Here's the change. Instead of having a fixed-sized output, the output size is based on the length of x_variable.
        output_distribution = self.fc_out(aggregated_representation).repeat(1, x_variable.shape[0])
        output_distribution = nn.functional.softmax(output_distribution, dim=1)
        
        return output_distribution
