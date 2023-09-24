import pickle

import torch

from algorithms.deep_mccfr_train import CFRNode
from algorithms.model import CitadelNetwork
from algorithms.train import train_model
from algorithms.visualization import visualize_cfr_tree
from game.game import Game

model = CitadelNetwork()
model.eval()

for _ in range(10):
    #try:
    game = Game()
    game.setup_round()
    winner = False
    position_root = CFRNode(game, original_player_id=0, model=model, role_pick_node=True)
    position_root.cfr(max_iterations=100000)
    targets = position_root.get_all_targets()
    train_model(targets, model, epochs=100, learning_rate=0.001, batch_size=64)
    #except Exception as e:
    #    print(e)
    #    continue

targets = []
for _ in range(10):
    try:
        game = Game()
        game.setup_round()
        winner = False
        position_root = CFRNode(game, original_player_id=0, model=model, role_pick_node=True)
        position_root.cfr(max_iterations=100000)
        targets += position_root.get_all_targets()

    except Exception as e:
        print(e)
        continue

with open("trainig_data.pkl", 'wb') as file:
    pickle.dump(targets, file)

train_model(targets, model, epochs=1000, learning_rate=0.001, batch_size=64)
torch.save(model.state_dict(), "model.pt")


