from game.game import Game

from random import choice
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.visualization import visualize_cfr_tree
from algorithms.models import VariableInputNN
import torch


i = 0

model = VariableInputNN()
#model.load_state_dict(torch.load("epoch_long1.pt"))
model.eval()
winners = [0, 0, 0, 0, 0, 0]
for _ in range(100):
    game = Game()
    game.setup_round()
    winner = False
    while not winner:
        i+=1
        if i % 100 == 0:
            position_root = CFRNode(game, original_player_id=game.gamestate.player_id)
            position_root.cfr_train()
            position_root.draw_gradients()
            position_root.children[0][1].draw_gradients(name="child.png")
            _, chosen_option = position_root.action_choice()
            winner = chosen_option.carry_out(game)
            1/0

        else:
            options = game.get_options_from_state()
            chosen_option = choice(options)
            winner = chosen_option.carry_out(game)
    print(f"{winner.id} won!")
    winners[winner.id] += 1

print(winners)
print(winner.id)