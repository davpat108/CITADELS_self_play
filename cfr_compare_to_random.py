from game.game import Game

from random import choice
from algorithms.deep_mccfr_train_masked_linear import CFRNode
from algorithms.visualization import visualize_cfr_tree
from algorithms.models import CitadelNetwork
import torch


i = 0
total_options = 0

model = CitadelNetwork()
model.load_state_dict(torch.load("model.pt"))
model.eval()
model.to("cpu")
winners = [0, 0, 0, 0, 0, 0]
for _ in range(100):
    game = Game()
    game.setup_round()
    winner = False
    try:
        while not winner:
            i+=1
            if game.gamestate.player_id  == 0:
                position_root = CFRNode(game, original_player_id=0, model=model)
                position_root.cfr(max_iterations=10000)
                _, chosen_option = position_root.action_choice()
                winner = chosen_option.carry_out(game)

            else:
                options,_ = game.get_options_from_state()
                options = [option for option_list in options.values() for option in option_list]
                total_options += len(options)
                chosen_option = choice(options)
                winner = chosen_option.carry_out(game)
        print(f"{winner.id} won!")
        winners[winner.id] += 1
    except Exception as e:
        print(e)
        continue

print(winners)
print(winner.id)