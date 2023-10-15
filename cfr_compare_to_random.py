from game.game import Game

from random import choice
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.visualization import visualize_cfr_tree
from algorithms.models import VariableInputNN
import torch


i = 0
game_encoding_size = 478
embedding_size = 10
num_heads = 3
num_transformer_layers = 2
vector_input_size = 5
model = VariableInputNN(game_encoding_size=game_encoding_size, vector_input_size=vector_input_size, embedding_size=embedding_size,
                        num_heads=num_heads, num_transformer_layers=num_transformer_layers)
model.load_state_dict(torch.load("epoch_long1.pt"))
model.eval()
winners = [0, 0, 0, 0, 0, 0]
for _ in range(100):
    game = Game()
    game.setup_round()
    winner = False
    while not winner:
        i+=1
        if game.gamestate.player_id  == 0:
            position_root = CFRNode(game, original_player_id=0, model=model)
            position_root.cfr_pred()
            _, chosen_option = position_root.action_choice()
            winner = chosen_option.carry_out(game)

        else:
            options = game.get_options_from_state()
            chosen_option = choice(options)
            winner = chosen_option.carry_out(game)
    print(f"{winner.id} won!")
    winners[winner.id] += 1

print(winners)
print(winner.id)