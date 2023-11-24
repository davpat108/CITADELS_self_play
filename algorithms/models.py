import torch.nn as nn
import torch


class VariableInputNN(nn.Module):
    def __init__(self, game_encoding_size=418, embedding_size=256, vector_input_size=131, num_heads=4, num_transformer_layers=2):
        super(VariableInputNN, self).__init__()
        
        # Embedding for the fixed input to match variable item size
        self.embedding_fixed = nn.Linear(game_encoding_size, embedding_size)
        
        # Embedding for the variable input
        self.embedding_variable = nn.Linear(vector_input_size, embedding_size)
        
        # Transformer layers
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=num_heads, num_encoder_layers=num_transformer_layers, batch_first=True)
        
        self.aggregate = nn.AdaptiveAvgPool1d(1)  # Using average pooling for aggregation
        # Final layers to produce scores
        self.fc_out_variable = nn.Linear(embedding_size, 1)
        self.fc_out_fixed = nn.Linear(embedding_size, 6)

    def forward(self, x_fixed, x_variable):
        N = x_variable.size(1) # [batch, N, dim]
        
        # Embed the fixed input
        x_fixed_embedded = self.embedding_fixed(x_fixed).unsqueeze(1) # [batch, 1, dim]
        
        # Embed the variable input
        x_variable_embedded = self.embedding_variable(x_variable) # [batch, N, dim]
        
        x_fixed_repeated = x_fixed_embedded.repeat(1, N, 1) # [batch, N, dim]
        transformer_output = self.transformer(x_variable_embedded, x_fixed_repeated) # [batch, N, dim]
        
        aggregated_representation = self.aggregate(transformer_output.permute(0, 2, 1)).squeeze(-1) # [batch, dim]
        
        output_distribution_variable = self.fc_out_variable(transformer_output).squeeze(-1)
        output_distribution_fixed = self.fc_out_fixed(aggregated_representation)
        
        return output_distribution_variable, output_distribution_fixed
