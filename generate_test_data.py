import pickle
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr import CFRNode

from game.game import Game
from multiprocessing import Pool, cpu_count
import os
from copy import deepcopy
import numpy as np
from tqdm import tqdm

def setup_game(game_index):
    try:
        move_stop_num = randint(1, 30)
        game = Game(debug=False)
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
        options = []
        limit = 0
        while len(options) < 2 and limit < 100:
            almost_won_game = games[-move_stop_num]
            options = almost_won_game.get_options_from_state()
            move_stop_num -= 1
            limit += 1

        
        #Encoding the game
        game_input = almost_won_game.encode_game()
        position_root = CFRNode(almost_won_game, original_player_id=almost_won_game.gamestate.player_id, model=None, training=True, device="cuda:0")
        position_root.cfr_train(max_iterations=20000)

        options = [option for option, _ in position_root.children]
        if len(options) == 0:
            return []
        options_input = torch.cat([option.encode_option() for option in options], dim=0).unsqueeze(0)

        if len(position_root.cumulative_regrets) == 0:
            return []
        target_decision_dist = torch.tensor(position_root.cumulative_regrets)
        
        # Handle no ragrets, and rolepick
        if target_decision_dist.size() == torch.Size([6, 10]):
            target_decision_dist=target_decision_dist[randint(0,5)]
        if torch.sum(target_decision_dist) == 0:
            target_decision_dist = torch.ones_like(target_decision_dist)
        
        return [(game_input, options_input, torch.tensor(position_root.node_value), target_decision_dist)]
    
    except ValueError:
        return []

def parallel_simulations(num_simulations):
    with Pool(cpu_count()-6) as pool:
        results = pool.starmap(setup_game, [(i,) for i in range(num_simulations)])
    return [item for sublist in results for item in sublist]


if __name__ == "__main__":
    val_targets = []
    for _ in tqdm(range(20), desc='Processing Validation Targets'):
        val_targets += parallel_simulations(30)
    with open(f"validation_targets.pkl", 'wb') as file:
        pickle.dump(val_targets, file)

    test_targets = []
    for _ in tqdm(range(20), desc='Processing Test Targets'):
        test_targets += parallel_simulations(30)
    with open(f"test_targets.pkl", 'wb') as file:
        pickle.dump(test_targets, file)