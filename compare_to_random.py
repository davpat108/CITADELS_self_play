import time

from random import choice
from multiprocessing import Pool

from run_utils import setup_model_for_eval, create_game, run_mccfr

def play_games(sim_number):
    
    model = setup_model_for_eval()
    winners = [0, 0, 0, 0, 0, 0]
    
    for _ in range(sim_number):
        game = create_game()
        winner = False
        
        while not winner:
            if game.gamestate.player_id == 0 and len(game.get_options_from_state()) > 1:
                start_time = time.time()
                chosen_option, _ = run_mccfr(game, model, max_iterations=200)
                winner = chosen_option.carry_out(game)
                print(f"decision for mccfr with model took: {time.time() - start_time}")
                
            elif game.gamestate.player_id == 1 and len(game.get_options_from_state()) > 1:
                start_time = time.time()
                chosen_option, _ = run_mccfr(game)
                winner = chosen_option.carry_out(game)
                
            else:
                options = game.get_options_from_state()
                chosen_option = choice(options)
                winner = chosen_option.carry_out(game)
                
        print(f"{winner.id} won!")
        winners[winner.id] += 1

    return winners

def parallel_simulations(num_simulations, sim_number_per_session=20):
    with Pool(num_simulations) as pool:
        results = pool.map(play_games, [sim_number_per_session for i in range(num_simulations)])
    return results

if __name__ == "__main__":
    print(parallel_simulations(5, 20))