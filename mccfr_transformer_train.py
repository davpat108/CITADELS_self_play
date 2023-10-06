import pickle

import torch

from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game

# Hyperparameter
game_encoding_size = 478
embedding_size = 10
num_heads = 3
num_transformer_layers = 2
vector_input_size = 5
model = VariableInputNN(game_encoding_size=game_encoding_size, vector_input_size=vector_input_size, embedding_size=embedding_size, 
                        num_heads=num_heads, num_transformer_layers=num_transformer_layers)
model.eval()

for _ in range(10):
    model.to("cpu")
    game = Game()
    game.setup_round()
    winner = False
    position_root = CFRNode(game, original_player_id=0, model=model, role_pick_node=True, training=True)
    position_root.cfr(max_iterations=1000)
    targets = position_root.get_all_targets()
    train_transformer(targets, model, epochs=50, lr=0.001, batch_size=1024)

targets = []
for _ in range(10):
    try:
        model.to("cpu")
        game = Game()
        game.setup_round()
        winner = False
        position_root = CFRNode(game, original_player_id=0, model=model, role_pick_node=True, training=True)
        position_root.cfr(max_iterations=100000)
        targets += position_root.get_all_targets()

    except Exception as e:
        print(e)
        continue

with open("trainig_data.pkl", 'wb') as file:
    pickle.dump(targets, file)

train_transformer(targets, model, epochs=50, lr=0.001, batch_size=1024)
torch.save(model.state_dict(), "model.pt")



