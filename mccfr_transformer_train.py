import pickle
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game
from multiprocessing import Pool, cpu_count
from algorithms.train_utils import draw_eval_results, draw_length_results, get_nodes_with_usefulness_treshold
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


def simulate_game(model, process_index, usefulness_treshold, pretrain=False):

    targets = []
    game = None
    while not game:
        game = setup_game(500)

    position_root = CFRNode(game, original_player_id=0, model=model if not pretrain else None, role_pick_node=game.gamestate.state==0, training=True, device="cuda:0")
    position_root.cfr_train(max_iterations=200000)
    targets += position_root.get_all_targets(usefulness_treshold=usefulness_treshold)
    print(f"Created {len(targets)} targets for training")
    print(f"Process {process_index} finished")

    return targets


def parallel_simulations(num_simulations, model, base_usefullness_treshold, pretrain=False):
    with Pool(cpu_count()) as pool:
        results = pool.starmap(simulate_game, [(model, i, base_usefullness_treshold, pretrain) for i in range(num_simulations)])
    return [item for sublist in results for item in sublist]


def get_mccfr_targets(model, minimum_sufficient_nodes, base_usefullness_treshold, pretrain=False):
    """
    args:
        model: the model to pretrain
        minimum_sufficient_nodes: the number of nodes to pretrainq
    Train the model with plain mccfr
    """
    targets = []
    while len(targets) < minimum_sufficient_nodes:
        try:
            targets += parallel_simulations(4, model, base_usefullness_treshold=base_usefullness_treshold, pretrain=pretrain)
            print(f"Total targets: {len(targets)}")
        except MemoryError:
            print("Memory Error")
            continue
    return targets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting training")
    model = VariableInputNN(game_encoding_size=478, fixed_embedding_size=256, variable_embedding_size=256, vector_input_size=131, num_heads=4, num_transformer_layers=2)
    model.eval()
    base_usefullness_treshold = 50
    max_usefullness_theshold = 200
    # pretrain
    if not f"10k_50thresh_pretrain.pkl" in os.listdir():
        #targets = simulate_game(model, 0, 0, pretrain=True)

        targets = get_mccfr_targets(model, minimum_sufficient_nodes=10000, base_usefullness_treshold=base_usefullness_treshold, pretrain=True)
        with open(f"10k_50thresh_pretrain.pkl", 'wb') as file:
            pickle.dump(targets, file)


        results = []
        lengths = []
        
        for i in range(base_usefullness_treshold, max_usefullness_theshold):
            sub_targets = get_nodes_with_usefulness_treshold(targets, i)
            if len(sub_targets) == 0:
                print(f"Usefulness treshold {i} has no targets")
                max_usefullness_theshold = i
                break
            
            results += train_transformer(sub_targets, model, epochs=75, batch_size=256, best_model_name=f"best_pretrain_model{i}.pt", verbose=False)
            lengths.append(len(sub_targets))
            
        best_index = results.index(min(results)) + base_usefullness_treshold
        os.rename(f"best_pretrain_model{best_index}.pt", f"best_pretrain_model.pt")
        
        draw_eval_results(results, base_usefullness_treshold, max_usefullness_theshold, name="pretrain_loss_plot.png")
        draw_length_results(lengths, base_usefullness_treshold, max_usefullness_theshold, name="pretrain_length_plot.png")

    # train 
    for u in range(3):
        if not f"10k_50thresh_train_{u}.pkl" in os.listdir():
            model.load_state_dict(torch.load(f"best_pretrain_model.pt"))
            targets = get_mccfr_targets(model, minimum_sufficient_nodes=10000, base_usefullness_treshold=base_usefullness_treshold)
            
            with open(f"10k_50thresh_train_{u}.pkl", 'wb') as file:
                pickle.dump(targets, file)
                
            results = []
            lengths = []
            
            for i in range(base_usefullness_treshold, max_usefullness_theshold):
                sub_targets = get_nodes_with_usefulness_treshold(targets, i)
                if len(sub_targets) == 0:
                    print(f"Usefulness treshold {i} has no targets")
                    max_usefullness_theshold = i
                    break
                
                results += train_transformer(sub_targets, model, epochs=75, best_model_name=f"best_train{u}_model{i}.pt", batch_size=256, verbose=False)
                lengths.append(len(sub_targets))
                
            best_index = results.index(min(results))+base_usefullness_treshold
            os.rename(f"best_train{u}_model{best_index}.pt", f"best_from_train_{u}.pt")
            
            draw_eval_results(results, base_usefullness_treshold, max_usefullness_theshold, name=f"train{u}_loss_plot.png")
            draw_length_results(lengths, base_usefullness_treshold, max_usefullness_theshold, name=f"train{u}_length_plot.png")

    logging.shutdown()