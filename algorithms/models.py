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
from torch.nn import Transformer

class VariableInputNN(nn.Module):
    def __init__(self, game_encoding_size=478, embedding_size=10, vector_input_size=5, num_heads=3, num_transformer_layers=2):
        """
        Hidden size = """
        super(VariableInputNN, self).__init__()
        
        # Embedding for the fixed input to match variable item size
        self.embedding_fixed = nn.Linear(game_encoding_size, embedding_size)
        
        # Transformer layers
        self.transformer = nn.Transformer(d_model=embedding_size+vector_input_size, nhead=num_heads, num_encoder_layers=num_transformer_layers)
        self.aggregate = nn.AdaptiveAvgPool1d(1)  # Using average pooling for aggregation
        # Final layer to produce scores
        self.fc_out_variable = nn.Linear(embedding_size+vector_input_size, 1)
        self.fc_out_fixed = nn.Linear(embedding_size + vector_input_size, 6)

    def forward(self, x_fixed, x_variable):
        # Get the sequence length (N) from x_variable
        N = x_variable.size(1)
        
        # Embed the fixed input
        x_fixed_embedded = self.embedding_fixed(x_fixed).unsqueeze(1)  # shape: [batch_size, 1, variable_item_size]
        
        # Repeat the embedded fixed input for concatenation
        x_fixed_repeated = x_fixed_embedded.repeat(1, N, 1)  # shape: [batch_size, N, variable_item_size]
        
        # Concatenate the fixed and variable inputs
        combined_seq = torch.cat([x_fixed_repeated, x_variable], dim=-1)
        
        # The transformer expects input in the shape [seq_len, batch_size, embed_size]. Transpose appropriately.
        combined_seq = combined_seq.transpose(0, 1)  # shape: [N, batch_size, variable+fixed size]
        
        # Pass through the transformer
        transformer_output = self.transformer(combined_seq, combined_seq)  # shape: [N, batch_size,  variable+fixed size]

        aggregated_representation = self.aggregate(transformer_output.permute(1, 2, 0)).squeeze(-1)  # shape: [batch_size, variable+fixed size]
        
        # Create variable-length distribution
        output_distribution_variable = self.fc_out_variable(aggregated_representation).repeat(1, N)
        output_distribution_variable = nn.functional.softmax(output_distribution_variable, dim=1)

        # Create fixed-length distribution
        output_distribution_fixed = self.fc_out_fixed(aggregated_representation)
        output_distribution_fixed = nn.functional.softmax(output_distribution_fixed, dim=1)
        
        return output_distribution_variable, output_distribution_fixed