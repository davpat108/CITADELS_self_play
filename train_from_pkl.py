import pickle as pkl
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game
from multiprocessing import Pool, cpu_count
from algorithms.train_utils import draw_eval_results, draw_length_results, get_nodes_with_usefulness_treshold, plot_avg_regrets
import os
import glob

model = VariableInputNN(game_encoding_size=439, fixed_embedding_size=256, variable_embedding_size=256, vector_input_size=131, num_heads=4, num_transformer_layers=2)
#model.load_state_dict(torch.load("best_pretrain_model.pt"))
logging.basicConfig(level=logging.INFO)


combined_targets = []

# Iterate over each .pkl file and append its contents to the combined_targets list
pkl_files = [file for file in glob.glob('C:/repos/CITADELS_self_play/extra/*.pkl')]
for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as file:
        targets = pkl.load(file)
        combined_targets.extend(targets)
plot_avg_regrets(combined_targets, name=f"avg_regrets_pretrain.png")
results = []
results_train = []
lengths = []

base_usefullness_treshold = 20
max_usefullness_theshold = 200
for i in range(base_usefullness_treshold, max_usefullness_theshold):
    sub_targets = get_nodes_with_usefulness_treshold(combined_targets, i)
    if len(sub_targets) == 0:
        print(f"Usefulness treshold {i} has no targets")
        max_usefullness_theshold = i
        break
    
    with open(f"validation_targets.pkl", 'rb') as file:
        val_targets = pkl.load(file)
    
    model = VariableInputNN(game_encoding_size=439, fixed_embedding_size=256, variable_embedding_size=256, vector_input_size=131, num_heads=4, num_transformer_layers=2)
    eval_results, train_results = train_transformer(sub_targets, val_targets, model, epochs=75, best_model_name=f"best_train{0}_model{i}.pt", batch_size=256, verbose=True)
    results += eval_results
    results_train += train_results
    lengths.append(len(sub_targets))

best_index = results.index(min(results))+base_usefullness_treshold
os.rename(f"best_train{0}_model{best_index}.pt", f"best_from_train_{0}.pt")

draw_eval_results(results, base_usefullness_treshold, max_usefullness_theshold, name=f"eval{0}_loss_plot.png")
draw_eval_results(results_train, base_usefullness_treshold, max_usefullness_theshold, name=f"train{0}_loss_plot.png")
draw_length_results(lengths, base_usefullness_treshold, max_usefullness_theshold, name=f"train{0}_length_plot.png")

logging.shutdown()