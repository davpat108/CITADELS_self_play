from game.game import Game

from random import choice
from algorithms.mccfr import CFRNode
from algorithms.visualization import visualize_cfr_tree
from algorithms.models import CitadelNetwork
import torch

game = Game()
game.setup_round()
winner = False
i = 0
total_options = 0

model = CitadelNetwork()
model.load_state_dict(torch.load("model_old.pt"))
model.eval()
while not winner:
    i+=1
    if game.gamestate.player_id  == 0:
        position_root = CFRNode(game, current_player_id=0, original_player_id=0, model=model)
        position_root.cfr(iterations=10000)
        _, chosen_option = position_root.action_choice()
        winner = chosen_option.carry_out(game)

    else:
        options,_ = game.get_options_from_state()
        total_options += len(options)
        chosen_option = choice(options)
        winner = chosen_option.carry_out(game)

print(total_options/i)
print(winner.id)