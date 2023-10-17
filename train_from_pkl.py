import os
import torch
import timeit
import pickle as pkl
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game
from multiprocessing import Pool, cpu_count

game_encoding_size = 478
embedding_size = 10
num_heads = 3
num_transformer_layers = 2
vector_input_size = 5
model = VariableInputNN(game_encoding_size=game_encoding_size, vector_input_size=vector_input_size, embedding_size=embedding_size,
                        num_heads=num_heads, num_transformer_layers=num_transformer_layers)
model.load_state_dict(torch.load("epoch1.pt"))
files = os.listdir()

pkl_files = [file for file in files if file.endswith('.pkl')]

combined_targets = []

# Iterate over each .pkl file and append its contents to the combined_targets list
for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as file:
        targets = pkl.load(file)
        combined_targets.extend(targets)

# Now, you can call the train_transformer function with the combined targets
train_transformer(combined_targets, model, batch_size=64, epochs=15)
torch.save(model.state_dict(), "combined_epochs.pt")
