import pickle

import torch

from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_masked_model
from game.game import Game

# Hyperparameters
option_encoding_size = 5
hidden_size = 10
num_heads = 2
num_layers = 2
model = VariableInputNN(option_encoding_size=option_encoding_size, hidden_size=hidden_size, 
                        num_heads=num_heads, num_layers=num_layers)
model.eval()

for _ in range(10):
    model.to("cpu")
    game = Game()
    game.setup_round()
    winner = False
    position_root = CFRNode(game, original_player_id=0, model=model, role_pick_node=True)
    position_root.cfr(max_iterations=100000)
    targets = position_root.get_all_targets()
    train_masked_model(targets, model, epochs=50, learning_rate=0.001, batch_size=1024)

targets = []
for _ in range(10):
    try:
        model.to("cpu")
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

train_masked_model(targets, model, epochs=500, learning_rate=0.001, batch_size=1024)
torch.save(model.state_dict(), "model.pt")



