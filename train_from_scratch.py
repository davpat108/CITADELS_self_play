import pickle
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr import CFRNode
from algorithms.models import ValueOnlyNN
from algorithms.train import train_node_value_only
from game.game import Game
from multiprocessing import Pool, cpu_count
from algorithms.train_utils import plot_avg_regrets, RanOutOfMemory
import os
from copy import deepcopy
import sys
import matplotlib
sys.setrecursionlimit(5000)  
matplotlib.use('Agg') 

def setup_game(max_move_num):
    move_stop_num = randint(1, max_move_num)
    game = Game(debug=True)
    game.setup_round()
    games = []
    games.append(deepcopy(game))
    winner = False
    while not winner:
        options = game.get_options_from_state()
        chosen_option = choice(options)
        winner = chosen_option.carry_out(game)
        games.append(deepcopy(game))
    
    game = games[-move_stop_num]


    return game



def simulate_game(model, process_index, usefulness_treshold, pretrain=False, max_iterations=300000):

    targets = []
    game = None
    while not game:
        game = setup_game(100)

    position_root = CFRNode(game, original_player_id=game.gamestate.player_id, model=model if not pretrain else None, training=True, device="cuda:0")
    position_root.cfr_train(max_iterations=max_iterations)
    targets += position_root.get_all_targets(usefulness_treshold=usefulness_treshold)
    print(f"Created {len(targets)} targets for training")
    print(f"Process {process_index} finished")

    return targets


def parallel_simulations(num_simulations, model, base_usefullness_treshold, pretrain=False, max_iterations=200000):
    with Pool(cpu_count()) as pool:
        results = pool.starmap(simulate_game, [(model, i, base_usefullness_treshold, pretrain, max_iterations) for i in range(num_simulations)])
    return [item for sublist in results for item in sublist]


def get_mccfr_targets(model, minimum_sufficient_nodes, base_usefullness_treshold, pretrain=False, max_iterations=200000):
    """
    args:
        model: the model to pretrain
        minimum_sufficient_nodes: the number of nodes to pretrainq
    Train the model with plain mccfr
    """
    targets = []
    while len(targets) < minimum_sufficient_nodes:
        try:
            targets += parallel_simulations(4, model, base_usefullness_treshold=base_usefullness_treshold, pretrain=pretrain, max_iterations=max_iterations)
            print(f"Total targets: {len(targets)}/{minimum_sufficient_nodes}")
        except RanOutOfMemory:
            print("Memory Error")
            torch.cuda.empty_cache()
            continue
    return targets



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting training")
    model = ValueOnlyNN(418, hidden_size = 512)
    model.eval()
    usefullness_treshold = 200
    # pretrain
    if not f"10k_50thresh_pretrain.pkl" in os.listdir():
        print("Pretraining")

        targets = get_mccfr_targets(model, minimum_sufficient_nodes=20000, base_usefullness_treshold=usefullness_treshold, max_iterations=200000, pretrain=True)
        with open(f"10k_50thresh_pretrain.pkl", 'wb') as file:
            pickle.dump(targets, file)
        
        os.makedirs(f"pretrain", exist_ok=True)
        plot_avg_regrets(targets, name=f"pretrain/avg_regrets_pretrain.png")
        with open(f"validation_targets.pkl", 'rb') as file:
            val_targets = pickle.load(file)

        train_node_value_only(targets, val_targets, lr=0.3703517140136571, hidden_size=512, gamma=0.8851980333411889, epochs=1000, parent_folder=f"pretrain", batch_size=2048, verbose=False)

        
    # train
    for u in range(5):
        usefullness_treshold = 200
        if not f"10k_50thresh_train_{u}.pkl" in os.listdir():
            print(f"training {u}")
            model = ValueOnlyNN(418, hidden_size = 512)
            model.eval()
            modelname = f"train{u-1}/best_model.pt" if u > 0 else "pretrain/best_model.pt"
            model.load_state_dict(torch.load(modelname))
            targets = get_mccfr_targets(model, minimum_sufficient_nodes=5000, base_usefullness_treshold=usefullness_treshold, max_iterations=200000)

            with open(f"10k_50thresh_train_{u}.pkl", 'wb') as file:
                pickle.dump(targets, file)

            with open(f"validation_targets.pkl", 'rb') as file:
                val_targets = pickle.load(file)

            os.makedirs(f"train{u}", exist_ok=True)
            plot_avg_regrets(targets, name=f"train{u}/avg_regrets_train.png")

            train_node_value_only(targets, val_targets, lr=0.02, hidden_size=512, gamma=0.8851980333411889, epochs=1000, parent_folder=f"train{u}", batch_size=2048, verbose=False)


    logging.shutdown()
