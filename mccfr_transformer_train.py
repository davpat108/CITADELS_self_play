import pickle

import torch
import timeit
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game
from multiprocessing import Pool, cpu_count
# Hyperparameter

def simulate_game(model, process_index):
    
    targets = []
    game = Game()
    game.setup_round()
    position_root = CFRNode(game, original_player_id=0, model=model, role_pick_node=True, training=True, device="cuda:0")
    position_root.cfr_train(max_iterations=50000)
    targets += position_root.get_all_targets()
    print(f"Process {process_index} finished")
    return targets


def parallel_simulations(num_simulations, model):
    with Pool(cpu_count()-1) as pool:
        results = pool.starmap(simulate_game, [(model, i) for i in range(num_simulations)])
    return [item for sublist in results for item in sublist]

if __name__ == "__main__":
    game_encoding_size = 478
    embedding_size = 10
    num_heads = 3
    num_transformer_layers = 2
    vector_input_size = 5
    model = VariableInputNN(game_encoding_size=game_encoding_size, vector_input_size=vector_input_size, embedding_size=embedding_size,
                            num_heads=num_heads, num_transformer_layers=num_transformer_layers)
    model.eval()
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for i in range(10):
        #
        #timer = timeit.Timer(lambda: parallel_simulations(12, model))
        #execution_time = timer.timeit(1)
        #print(f"Execution time: {execution_time}")
        #raise
        #targets = simulate_game(model, 0)
        targets = parallel_simulations(4, model)

        print("Finished simulations")

        with open(f"trainig_data_long{i}.pkl", 'wb') as file:
            pickle.dump(targets, file)

        train_transformer(targets, model, epochs=15, batch_size=6)
        torch.save(model.state_dict(), f"epoch_long{i}.pt")