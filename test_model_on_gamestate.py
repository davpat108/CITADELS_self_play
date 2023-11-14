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

def setup_game():
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


    almost_won_game = games[-20]
    result  = np.zeros(6)
    result[winner.id] = 1
    print(f"Winner: {winner.id}, so node value should be {result}")
    return almost_won_game


def predict_game(model, game):

    targets = []
    game = None
    while not game:
        game = setup_game(500)

    position_root = CFRNode(game, original_player_id=0, model=model if not pretrain else None, role_pick_node=game.gamestate.state==0, training=True, device="cuda:0")
    position_root.cfr_train(max_iterations=max_iterations)
    targets += position_root.get_all_targets(usefulness_treshold=usefulness_treshold)
    print(f"Created {len(targets)} targets for training")
    print(f"Process {process_index} finished")

    return targets



model_config = {
    'game_encoding_size': 478,
    'fixed_embedding_size': 256,
    'variable_embedding_size': 256,
    'vector_input_size': 131,
    'num_heads': 4,
    'num_transformer_layers': 2
}

if __name__ == "__main__":
    print("Starting training")
    model = VariableInputNN(**model_config)
    model.load_state_dict(torch.load("train4/best_from_train.pt"))
    model.eval()
    
    game = setup_game()
    game_input = game.encode_game().unsqueeze(0)
    options = game.get_options_from_state()
    options_input = torch.cat([option.encode_option() for option in options], dim=0).unsqueeze(0)
    distribution, node_value = model(game_input, options_input)
    winning_probabilities = square_and_normalize(node_value, dim=1).squeeze(0).detach().numpy()
    print(f"Winning probabilities: {winning_probabilities}")
    
    position_root = CFRNode(game, original_player_id=game.gamestate.player_id, model=None, role_pick_node=game.gamestate.state==0, training=True, device="cuda:0")
    position_root.cfr_train(max_iterations=10000)
    print(f"value: {position_root.node_value}, strategy: {position_root.strategy}")
    