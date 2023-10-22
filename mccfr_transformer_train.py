import pickle
from random import choice, randint
import torch
import logging
from algorithms.deep_mccfr_transformer import CFRNode
from algorithms.models import VariableInputNN
from algorithms.train import train_transformer
from game.game import Game
from multiprocessing import Pool, cpu_count
import seaborn as sns
import matplotlib.pyplot as plt
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
        logging.info(f"Game created with {move_stop_num} moves in")
        return game
    else:
        logging.info("Accidental winner")
        return None


def simulate_game(model, process_index, usefulness_treshold, pretrain=False):

    targets = []
    game = None
    while not game:
        game = setup_game(500)
    
    position_root = CFRNode(game, original_player_id=0, model=model if not pretrain else None, role_pick_node=game.gamestate.state==0, training=True, device="cuda:0")
    position_root.cfr_train(max_iterations=60000)
    targets += position_root.get_all_targets(usefulness_treshold=usefulness_treshold)
    logging.info(f"Created {len(targets)} targets for training")
    logging.info(f"Process {process_index} finished")

    return targets


def parallel_simulations(num_simulations, model, base_usefullness_treshold, pretrain=False):
    with Pool(cpu_count()) as pool:
        results = pool.starmap(simulate_game, [(model, i, base_usefullness_treshold, pretrain) for i in range(num_simulations)])
    return [item for sublist in results for item in sublist]

def set_logger():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('output.txt')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Adjust the log level as needed

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def draw_eval_results(results, base_usefullness_treshold, max_usefullness_theshold):
    x_values = list(range(base_usefullness_treshold, max_usefullness_theshold))
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.lineplot(x=x_values, y=results)
    plt.xlabel("Usefulness Treshold")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss vs Usefulness Treshold")
    plt.savefig("loss_plot.png")
    plt.close()


def draw_length_results(lengths, base_usefullness_treshold, max_usefullness_theshold):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.lineplot(x=list(range(base_usefullness_treshold, max_usefullness_theshold)), y=lengths)
    plt.xlabel("Usefulness Treshold")
    plt.ylabel("Number of Targets")
    plt.title("Number of Targets vs Usefulness Treshold")
    plt.savefig("length_plot.png")
    plt.close()


def get_nodes_with_usefulness_treshold(targets, treshold):
    """
    args:
        targets: list of targets
        treshold: the minimum usefulness of a target
    returns:
        list of targets with usefulness >= treshold
    """
    return [target for target in targets if target[3].sum() >= treshold]


def get_plain_mccfr_targets(model, minimum_sufficient_nodes, base_usefullness_treshold):
    """
    args:
        model: the model to pretrain
        minimum_sufficient_nodes: the number of nodes to pretrainq
    Train the model with plain mccfr
    """
    targets = []
    while len(targets) < minimum_sufficient_nodes:
        targets += parallel_simulations(12, model, base_usefullness_treshold=base_usefullness_treshold, pretrain=True)
    return targets


if __name__ == "__main__":
    set_logger()
    model = VariableInputNN()
    model.eval()
    base_usefullness_treshold = 10
    max_usefullness_theshold = 10
    targets = get_plain_mccfr_targets(model, minimum_sufficient_nodes=10000, base_usefullness_treshold=base_usefullness_treshold)
    with open(f"10k.pkl", 'wb') as file:
        pickle.dump(targets, file)

    results = []
    lengths = []

    for i in range(0, max_usefullness_theshold):
        sub_targets = get_nodes_with_usefulness_treshold(targets, i)
        if len(sub_targets) == 0:
            logging.info(f"Usefulness treshold {i} has no targets")
            max_usefullness_theshold = i
            break
        results += train_transformer(sub_targets, model, epochs=75, batch_size=64)
        lengths.append(len(sub_targets))
    draw_eval_results(results, base_usefullness_treshold, max_usefullness_theshold)
    draw_length_results(lengths, base_usefullness_treshold, max_usefullness_theshold)

    logging.shutdown()