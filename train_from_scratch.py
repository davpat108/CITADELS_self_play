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
import sys
import matplotlib
from run_utils import setup_model_for_eval, create_game, run_mccfr, create_a_random_game


sys.setrecursionlimit(5000)  
matplotlib.use('Agg') 




def simulate_game(model, process_index, usefulness_treshold, pretrain=False, max_iterations=300000):

    targets = []
    game = None
    while not game:
        game = create_a_random_game(100)

    _, position_root = run_mccfr(game, model=model if not pretrain else None, max_iterations=max_iterations, training=True)
    
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
        model: the model to use if not pratraining
        minimum_sufficient_nodes: the number of nodes to create
        base_usefullness_treshold: the amount of backprogations a nodes needs to be used
        pretrain: whether the model is used for decisionmaking in the mccfr run (so the first run is a pretrain)
        max_iterations: the maximum amount of iterations to run mccfr for
    Creates targets for the model to train on, attempts to handle memory errors
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
    model = ValueOnlyNN(418, hidden_size = 512)
    model.eval()
    usefullness_treshold = 200
    if not f"10k_50thresh_pretrain.pkl" in os.listdir():
        print("Model free data generation started")

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
            print(f"Data gen with already trained model nr. {u} started")
            model = setup_model_for_eval(f"train{u-1}/best_model.pt" if u > 0 else "pretrain/best_model.pt")
            targets = get_mccfr_targets(model, minimum_sufficient_nodes=5000, base_usefullness_treshold=usefullness_treshold, max_iterations=200000)

            with open(f"10k_50thresh_train_{u}.pkl", 'wb') as file:
                pickle.dump(targets, file)

            with open(f"validation_targets.pkl", 'rb') as file:
                val_targets = pickle.load(file)

            os.makedirs(f"train{u}", exist_ok=True)
            plot_avg_regrets(targets, name=f"train{u}/avg_regrets_train.png")

            train_node_value_only(targets, val_targets, lr=0.02, hidden_size=512, gamma=0.8851980333411889, epochs=1000, parent_folder=f"train{u}", batch_size=2048, verbose=False)


    logging.shutdown()
