import torch.nn as nn
import torch


class VariableInputNN(nn.Module):
    def __init__(self, game_encoding_size=478, fixed_embedding_size=100, variable_embedding_size=100, vector_input_size=131, num_heads=4, num_transformer_layers=2):
        super(VariableInputNN, self).__init__()
        
        # Embedding for the fixed input to match variable item size
        self.embedding_fixed = nn.Linear(game_encoding_size, fixed_embedding_size)
        
        # Embedding for the variable input
        self.embedding_variable = nn.Linear(vector_input_size, variable_embedding_size)
        
        # Transformer layers
        transformer_input_size = fixed_embedding_size + variable_embedding_size
        self.transformer = nn.Transformer(d_model=transformer_input_size, nhead=num_heads, num_encoder_layers=num_transformer_layers)
        
        self.aggregate = nn.AdaptiveAvgPool1d(1)  # Using average pooling for aggregation
        # Final layers to produce scores
        self.fc_out_variable = nn.Linear(transformer_input_size, 1)
        self.fc_out_fixed = nn.Linear(transformer_input_size, 6)

    def forward(self, x_fixed, x_variable):
        N = x_variable.size(1)
        
        # Embed the fixed input
        x_fixed_embedded = self.embedding_fixed(x_fixed).unsqueeze(1)
        
        # Embed the variable input
        x_variable_embedded = self.embedding_variable(x_variable)
        
        x_fixed_repeated = x_fixed_embedded.repeat(1, N, 1)
        combined_seq = torch.cat([x_fixed_repeated, x_variable_embedded], dim=-1)
        
        combined_seq = combined_seq.transpose(0, 1)
        transformer_output = self.transformer(combined_seq, combined_seq)
        
        aggregated_representation = self.aggregate(transformer_output.permute(1, 2, 0)).squeeze(-1)
        
        output_distribution_variable = self.fc_out_variable(transformer_output.permute(1, 0, 2)).squeeze(-1)
        output_distribution_fixed = self.fc_out_fixed(aggregated_representation)
        
        return output_distribution_variable, output_distribution_fixed
