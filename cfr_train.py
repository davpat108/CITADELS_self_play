from game.game import Game

from random import choice
from algorithms.deep_mccfr_train import CFRNode
from algorithms.visualization import visualize_cfr_tree
from algorithms.model import CitadelNetwork
import torch
from algorithms.train import train_model

model = CitadelNetwork()
model.eval()

for _ in range(10):
    game = Game()
    game.setup_round()
    winner = False
    position_root = CFRNode(game, current_player_id=0, original_player_id=0, model=model)
    position_root.cfr(max_iterations=10)
    targets = position_root.get_all_targets()
    train_model(targets, model, epochs=10, learning_rate=0.001, batch_size=64)





print(winner.id)