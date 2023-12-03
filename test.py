import pickle as pkl
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN, ValueOnlyNN
from algorithms.train import train_transformer, train_node_value_only
from game.game import Game
from multiprocessing import Pool, cpu_count
from algorithms.train_utils import draw_eval_results, draw_length_results, get_nodes_with_usefulness_treshold, plot_avg_regrets
import os
import glob

logging.basicConfig(level=logging.INFO)





with open(f"10k_50thresh_pretrain.pkl", 'rb') as file:
    targets = pkl.load(file)

with open(f"validation_targets.pkl", 'rb') as file:
    val_targets = pkl.load(file)

# {'gamma': 0.8851980333411889, 'hidden_size': 2, 'lr': 0.3703517140136571}
# Call your training function
loss = train_node_value_only(targets, val_targets, lr=0.3703517140136571, hidden_size=512, gamma=0.8851980333411889, epochs=1000, parent_folder=f"pretrain_final_test", batch_size=2048, verbose=True)
    


logging.shutdown()