import pickle
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game
from multiprocessing import Pool, cpu_count
from algorithms.train_utils import draw_eval_results, draw_length_results, get_nodes_with_usefulness_treshold, plot_avg_regrets, RanOutOfMemory, square_and_normalize
import os
from copy import deepcopy
import numpy as np
from tqdm import tqdm

def setup_game(game_index):
    try:
        #Running the game
        #print(f"Starting game {game_index}")
        move_stop_num = randint(1, 30)
        game = Game()
        game.setup_round()
        winner = False
        tota_options = 0
        games = []
        games.append(deepcopy(game))
        while not winner:
            options = game.get_options_from_state()
            tota_options += len(options)
            chosen_option = choice(options)
            winner = chosen_option.carry_out(game)
            games.append(deepcopy(game))

        #Picking a game
        almost_won_game = games[-move_stop_num]
        result  = np.zeros(6)
        result[winner.id] = 1
        #print(f"Winner: {winner.id}, so node value should be {result}")
        
        #Encoding the game
        game_input = almost_won_game.encode_game()
        position_root = CFRNode(almost_won_game, original_player_id=almost_won_game.gamestate.player_id, model=None, role_pick_node=almost_won_game.gamestate.state==0, training=True, device="cuda:0")
        position_root.cfr_train(max_iterations=20000)

        options = [option for option, _ in position_root.children]
        if len(options) == 0:
            return []
        options_input = torch.cat([option.encode_option() for option in options], dim=0).unsqueeze(0)

        if len(position_root.cumulative_regrets) == 0:
            return []
        target_decision_dist = torch.tensor(position_root.cumulative_regrets)
        
        # Handle no ragrets, and rolepick
        if torch.sum(target_decision_dist) == 0:
            target_decision_dist = torch.ones_like(target_decision_dist)
        if target_decision_dist.size() == torch.Size([6, 10]):
            target_decision_dist=target_decision_dist[almost_won_game.gamestate.player_id]
        
        return [(game_input, options_input, torch.tensor(position_root.node_value), target_decision_dist)]
    
    except ValueError:
        return []

def parallel_simulations(num_simulations):
    with Pool(cpu_count()-1) as pool:
        results = pool.starmap(setup_game, [(i,) for i in range(num_simulations)])
    return [item for sublist in results for item in sublist]


if __name__ == "__main__":
    val_targets = []
    for _ in tqdm(range(20), desc='Processing Validation Targets'):
        val_targets += parallel_simulations(33)
    with open(f"validation_targets.pkl", 'wb') as file:
        pickle.dump(val_targets, file)

    test_targets = []
    for _ in tqdm(range(20), desc='Processing Test Targets'):
        test_targets += parallel_simulations(33)
    with open(f"test_targets.pkl", 'wb') as file:
        pickle.dump(test_targets, file)