from game.game import Game

from random import choice
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.visualization import visualize_cfr_tree
import pickle

game = Game()
game.setup_round()
winner = False
i = 0
tota_options = 0
while not winner:
    i+=1
    if game.gamestate.player_id  == 0:
        position_root = CFRNode(game, original_player_id=0)
        position_root.cfr_train(max_iterations=100000)
        _, chosen_option = position_root.action_choice()
        print(chosen_option.name)
        winner = chosen_option.carry_out(game)

    else:
        options = game.get_options_from_state()
        tota_options += len(options)
        chosen_option = choice(options)
        print(chosen_option.name)
        winner = chosen_option.carry_out(game)

print(tota_options/i)
print(winner.id)