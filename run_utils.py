import torch

from random import choice, randint
from copy import deepcopy

from algorithms.models import ValueOnlyNN
from game.game import Game
from algorithms.deep_mccfr import CFRNode


def setup_model_for_eval(model_path="pretrain/best_model.pt"):
    """
    Returns a pre-trained model for evaluation
    """
    model = ValueOnlyNN(418, hidden_size=512)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def create_game():
    """
    Returns an already setup game object
    """
    game = Game(preset=True)
    game.setup_round()
    
    return game

def create_a_close_to_finished_game(game):
    """
    Returns a close to finished gamestate
    """
    winner = False
    games = []
    move_stop_num = randint(1, 30)
    games.append(deepcopy(game))
    while not winner:
        options = game.get_options_from_state()
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
        
    return almost_won_game

def create_a_random_game(max_move_num):
    """
    Create a gme with max_move_num random moves in
    """
    move_stop_num = randint(1, max_move_num)
    game = create_game()
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

def run_mccfr(game, model=None, max_iterations=2000, training=False):
    """
    Uses the model to make a decision in the game, if no model is provided, regular MCCFR is used
    """
    if model and not training:
        position_root_model = CFRNode(game, original_player_id=game.gamestate.player_id, model=model, training=training)
        position_root_model.cfr_pred(max_iterations=max_iterations, max_depth=10)
        _, chosen_option = position_root_model.action_choice(live=True)
    else:
        position_root_model = CFRNode(game, original_player_id=game.gamestate.player_id, training=training, model=model)
        position_root_model.cfr_train(max_iterations=max_iterations)
        _, chosen_option = position_root_model.action_choice(live=True)
        
    return chosen_option, position_root_model

def encode_options_from_node(node):
    """
    Encodes the options from a node
    """
    options = [option for option, _ in node.children]
    if len(options) == 0:
        return []
    options_input = torch.cat([option.encode_option() for option in options], dim=0).unsqueeze(0)
    return options_input

def create_target_strategy(node):
    """
    Creates a target strategy from a node
    """
    target_decision_dist = torch.tensor(node.cumulative_regrets)
    # Handle no ragrets, and rolepick
    if target_decision_dist.size() == torch.Size([6, 10]):
        target_decision_dist=target_decision_dist[randint(0,5)]
    if torch.sum(target_decision_dist) == 0:
        target_decision_dist = torch.ones_like(target_decision_dist)
    return target_decision_dist