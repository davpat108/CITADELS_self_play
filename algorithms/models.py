import torch.nn as nn
import torch
import torch.nn.functional as F

class VariableInputNN(nn.Module):
    def __init__(self, game_encoding_size=418, embedding_size=256, vector_input_size=131, num_heads=4, num_transformer_layers=2, dropout_rate=0.1):
        super(VariableInputNN, self).__init__()
        
        # Embedding for the fixed input
        self.embedding_fixed = nn.Linear(game_encoding_size, embedding_size)
        self.dropout_fixed = nn.Dropout(dropout_rate)
        
        # Embedding for the variable input
        self.embedding_variable = nn.Linear(vector_input_size, embedding_size)
        self.dropout_variable = nn.Dropout(dropout_rate)
        
        # Transformer layers
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=num_heads, num_encoder_layers=num_transformer_layers, batch_first=True)
        
        # Pooling and final output layers
        self.aggregate = nn.AdaptiveAvgPool1d(1)
        self.fc_out_variable = nn.Linear(embedding_size, 1)
        self.fc_out_fixed = nn.Linear(embedding_size, 6)

    def forward(self, x_fixed, x_variable):
        N = x_variable.size(1)  # [batch, N, dim]
        
        # Process the fixed input
        x_fixed_embedded = self.embedding_fixed(x_fixed)  # [batch, dim]
        x_fixed_embedded = F.relu(x_fixed_embedded)
        x_fixed_embedded = self.dropout_fixed(x_fixed_embedded)
        x_fixed_embedded = x_fixed_embedded.unsqueeze(1)  # [batch, 1, dim]
        
        # Process the variable input
        x_variable_embedded = self.embedding_variable(x_variable)  # [batch, N, dim]
        x_variable_embedded = F.relu(x_variable_embedded)
        x_variable_embedded = self.dropout_variable(x_variable_embedded)
        
        # Transformer
        x_fixed_repeated = x_fixed_embedded.repeat(1, N, 1)  # [batch, N, dim]
        transformer_output = self.transformer(x_variable_embedded, x_fixed_repeated)  # [batch, N, dim]
        
        # Pooling and final output
        aggregated_representation = self.aggregate(transformer_output.permute(0, 2, 1)).squeeze(-1)  # [batch, dim]
        output_distribution_variable = self.fc_out_variable(transformer_output).squeeze(-1)
        output_distribution_fixed = self.fc_out_fixed(aggregated_representation)
        
        return output_distribution_variable, output_distribution_fixed


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