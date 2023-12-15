from game.game import Game

from random import choice
from algorithms.deep_mccfr import CFRNode
from algorithms.models import ValueOnlyNN
import torch
import time
from multiprocessing import Pool, cpu_count

def play_games(sim_number):
    model = ValueOnlyNN(418, hidden_size=512)
    model.load_state_dict(torch.load("pretrain/best_model.pt"))
    model.eval()
    winners = [0, 0, 0, 0, 0, 0]
    for _ in range(sim_number):
        game = Game(debug=True)
        game.setup_round()
        winner = False
        while not winner:
            if game.gamestate.player_id == 0 and len(game.get_options_from_state()) > 1:
                start_time = time.time()
                position_root_model = CFRNode(game, original_player_id=game.gamestate.player_id, model=model)
                position_root_model.cfr_pred(max_iterations=200, max_depth=10)
                #position_root_model.draw_gradients("modelled.png")
                #position_root_model.children[0][1].draw_gradients(name="modelled_child.png")
                _, chosen_option = position_root_model.action_choice(live=True)
                winner = chosen_option.carry_out(game)
                print(f"confidences: {position_root_model.cumulative_regrets}")
                print(f"decision took: {time.time() - start_time}")
                
            elif game.gamestate.player_id == 1 and len(game.get_options_from_state()) > 1:
                start_time = time.time()
                position_root_model = CFRNode(game, original_player_id=game.gamestate.player_id)
                position_root_model.cfr_train(max_iterations=2000)
                #position_root_model.draw_gradients("modelled.png")
                #position_root_model.children[0][1].draw_gradients(name="modelled_child.png")
                _, chosen_option = position_root_model.action_choice(live=True)
                winner = chosen_option.carry_out(game)
                print(f"confidences: {position_root_model.cumulative_regrets}")
                print(f"decision took: {time.time() - start_time}")
            else:
                options = game.get_options_from_state()
                chosen_option = choice(options)
                winner = chosen_option.carry_out(game)
            print(f"Player {chosen_option.attributes['perpetrator']} chose {chosen_option}")
            #used_options.append((chosen_option["perpetrator"], chosen_option))
        print(f"{winner.id} won!")
        winners[winner.id] += 1

    return winners

def parallel_simulations(num_simulations, sim_number_per_session=20):
    with Pool(num_simulations) as pool:
        results = pool.map(play_games, [sim_number_per_session for i in range(num_simulations)])
    return results

if __name__ == "__main__":
    print(parallel_simulations(5, 20))