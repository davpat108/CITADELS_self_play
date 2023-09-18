from game.game import Game

from random import choice
from algorithms.deep_mccfr_train import CFRNode
from algorithms.visualization import visualize_cfr_tree
from algorithms.model import CitadelNetwork
import torch

model = CitadelNetwork()
model.eval()

for _ in range(10):
    game = Game()
    game.setup_round()
    winner = False
    position_root = CFRNode(game, current_player_id=0, original_player_id=0, model=model)
    position_root.cfr()





print(winner.id)