import pickle
import torch

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from run_utils import create_game, create_a_close_to_finished_game, run_mccfr, encode_options_from_node, create_target_strategy

def setup_game(game_index):
    try:
        game = create_game()
        almost_won_game = create_a_close_to_finished_game(game)

        game_input = almost_won_game.encode_game()
        _, position_root = run_mccfr(game=almost_won_game, max_iterations=20000)

        options_input = encode_options_from_node(position_root)

        # Meaningless gamestate
        if len(position_root.cumulative_regrets) == 0:
            return []
        
        target_decision_dist = create_target_strategy(position_root)
        
        return [(game_input, options_input, torch.tensor(position_root.node_value), target_decision_dist)]
    
    except ValueError:
        return []

def parallel_simulations(num_simulations):
    with Pool(cpu_count()-6) as pool:
        results = pool.starmap(setup_game, [(i,) for i in range(num_simulations)])
    return [item for sublist in results for item in sublist]


if __name__ == "__main__":
    val_targets = []
    for _ in tqdm(range(20), desc='Processing Validation Targets'):
        val_targets += parallel_simulations(30)
    with open(f"validation_targets.pkl", 'wb') as file:
        pickle.dump(val_targets, file)

    test_targets = []
    for _ in tqdm(range(20), desc='Processing Test Targets'):
        test_targets += parallel_simulations(30)
    with open(f"test_targets.pkl", 'wb') as file:
        pickle.dump(test_targets, file)