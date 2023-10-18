import torch.nn as nn


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
        
        x_fixed_embedded = self.embedding_fixed(x_fixed).unsqueeze(1)  # shape: [batch_size, 1, variable_item_size]
        x_fixed_repeated = x_fixed_embedded.repeat(1, N, 1)  # shape: [batch_size, N, variable_item_size]
        combined_seq = torch.cat([x_fixed_repeated, x_variable], dim=-1)
        
        combined_seq = combined_seq.transpose(0, 1)  # shape: [N, batch_size, variable+fixed size]
        transformer_output = self.transformer(combined_seq, combined_seq)  # shape: [N, batch_size,  variable+fixed size]

        aggregated_representation = self.aggregate(transformer_output.permute(1, 2, 0)).squeeze(-1)  # shape: [batch_size, variable+fixed size]
        
        output_distribution_variable = self.fc_out_variable(transformer_output.permute(1, 0, 2))
        output_distribution_fixed = self.fc_out_fixed(aggregated_representation)
        
        return output_distribution_variable, output_distribution_fixed