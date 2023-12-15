import pickle
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr import CFRNode
from algorithms.models import ValueOnlyNN
from game.game import Game
from algorithms.train_utils import square_and_normalize
from copy import deepcopy
import numpy as np

def setup_game():
    game = Game(debug=True)
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





if __name__ == "__main__":
    print("Starting training")
    model0 = ValueOnlyNN(418, hidden_size=512)
    model1 = ValueOnlyNN(418, hidden_size=512)
    model2 = ValueOnlyNN(418, hidden_size=512)
    model3 = ValueOnlyNN(418, hidden_size=512)
    model4 = ValueOnlyNN(418, hidden_size=512)
    
    model0.load_state_dict(torch.load("pretrain/best_model.pt"))
    model1.load_state_dict(torch.load("train0/best_model.pt"))
    model2.load_state_dict(torch.load("train1/best_model.pt"))
    model3.load_state_dict(torch.load("train2/best_model.pt"))
    model4.load_state_dict(torch.load("train3/best_model.pt"))
    
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    
    game = setup_game()
    game_input = game.encode_game().unsqueeze(0)
    options = game.get_options_from_state()
    

    node_value = model0(game_input)
    winning_probabilities = square_and_normalize(node_value, dim=1).squeeze(0).detach().numpy()
    print(f"Winning probabilities model 1: {winning_probabilities}")
    node_value = model1(game_input)
    winning_probabilities = square_and_normalize(node_value, dim=1).squeeze(0).detach().numpy()
    print(f"Winning probabilities model 2: {winning_probabilities}")
    node_value = model2(game_input)
    winning_probabilities = square_and_normalize(node_value, dim=1).squeeze(0).detach().numpy()
    print(f"Winning probabilities model 3: {winning_probabilities}")
    node_value = model3(game_input)
    winning_probabilities = square_and_normalize(node_value, dim=1).squeeze(0).detach().numpy()
    print(f"Winning probabilities model 4: {winning_probabilities}")
    node_value = model4(game_input)
    winning_probabilities = square_and_normalize(node_value, dim=1).squeeze(0).detach().numpy()
    print(f"Winning probabilities model 5: {winning_probabilities}")
    
    
    
    position_root = CFRNode(game, original_player_id=game.gamestate.player_id, model=None, training=True, device="cuda:0")
    position_root.cfr_train(max_iterations=100000)
    print(f"value: {position_root.node_value}, strategy: {position_root.strategy}")
    print(options[0].name)
    print(options[1].name)
    