import pickle
from random import choice, randint
import torch
import timeit
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game
from multiprocessing import Pool, cpu_count
# Hyperparameter

def setup_game(max_move_num):
    move_stop_num = randint(1, max_move_num)
    game = Game()
    game.setup_round()
    winner = False
    i = 0
    tota_options = 0
    while not winner and i < move_stop_num:
        options = game.get_options_from_state()
        tota_options += len(options)
        chosen_option = choice(options)
        winner = chosen_option.carry_out(game)
        i+=1

    if not winner:
        print(f"Game created with {move_stop_num} moves in")
        return game
    else:
        print("Accidental winner")
        return None


def simulate_game(model, process_index, pretrain=False):

    targets = []
    game = None
    while not game:
        game = setup_game(500)
    
    position_root = CFRNode(game, original_player_id=0, model=model if not pretrain else None, role_pick_node=game.gamestate.state==0, training=True, device="cuda:0")
    position_root.cfr_train(max_iterations=200000)
    targets += position_root.get_all_targets(usefulness_treshold=15)
    print(f"Created {len(targets)} targets for training")
    print(f"Process {process_index} finished")

    return targets


def parallel_simulations(num_simulations, model, pretrain=False):
    with Pool(cpu_count()-1) as pool:
        results = pool.starmap(simulate_game, [(model, i, pretrain) for i in range(num_simulations)])
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
    for i in range(4):
        #targets = simulate_game(model, 0, pretrain=True)
        targets = parallel_simulations(4, model, pretrain=True)
        print("Finished simulations")

        with open(f"naked_mccfr_training_data{i}.pkl", 'wb') as file:
            pickle.dump(targets, file)

        train_transformer(targets, model, epochs=15, batch_size=6)
        torch.save(model.state_dict(), f"epoch{i}.pt")