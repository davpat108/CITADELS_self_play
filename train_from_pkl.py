import pickle as pkl
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game
from multiprocessing import Pool, cpu_count
from algorithms.train_utils import draw_eval_results, draw_length_results, get_nodes_with_usefulness_treshold
import os

model = VariableInputNN(game_encoding_size=478, fixed_embedding_size=256, variable_embedding_size=256, vector_input_size=131, num_heads=4, num_transformer_layers=2)
model.load_state_dict(torch.load("best_pretrain_model.pt"))
files = os.listdir()
logging.basicConfig(level=logging.INFO)
#pkl_files = [file for file in files if file.endswith('.pkl')]

combined_targets = []

# Iterate over each .pkl file and append its contents to the combined_targets list
#for pkl_file in pkl_files:
with open("10k_50thresh_train_0.pkl", 'rb') as file:
    targets = pkl.load(file)
    combined_targets.extend(targets)

results = []
lengths = []

base_usefullness_treshold = 30
max_usefullness_theshold = 200
for i in range(base_usefullness_treshold, max_usefullness_theshold):
    sub_targets = get_nodes_with_usefulness_treshold(targets, i)
    if len(sub_targets) == 0:
        print(f"Usefulness treshold {i} has no targets")
        max_usefullness_theshold = i
        break
    results += train_transformer(sub_targets, model, epochs=75, best_model_name=f"best_train{0}_model{i}.pt", batch_size=256, verbose=False)
    lengths.append(len(sub_targets))
best_index = results.index(min(results))+base_usefullness_treshold
os.rename(f"best_train{0}_model{best_index}.pt", f"best_from_train_{0}.pt")
            
draw_eval_results(results, base_usefullness_treshold, max_usefullness_theshold, name=f"train{0}_loss_plot.png")
draw_length_results(lengths, base_usefullness_treshold, max_usefullness_theshold, name=f"train{0}_length_plot.png")

logging.shutdown()