from game.game import Game

from random import choice
from algorithms.deep_mccfr import CFRNode
from algorithms.models import VariableInputNN, ValueOnlyNN
import torch


i = 0

model = ValueOnlyNN(418, hidden_size=512)
model.load_state_dict(torch.load("pretrain/best_model.pt"))
model.eval()
winners = [0, 0, 0, 0, 0, 0]
for _ in range(100):
    game = Game(debug=True)
    game.setup_round()
    winner = False
    while not winner:
        i+=1
        if i % 100 == 0 and len(game.get_options_from_state()) > 1:
            position_root_raw = CFRNode(game, original_player_id=game.gamestate.player_id)
            position_root_raw.cfr_train(max_iterations=200000)
            position_root_raw.draw_gradients()
            position_root_raw.children[0][1].draw_gradients(name="child.png")
            print(position_root_raw.cumulative_regrets)
            _, chosen_option = position_root_raw.action_choice(live=True)
            print([chosen_option.name for chosen_option, _ in position_root_raw.children])
            #winner = chosen_option.carry_out(game)
            #1/0
            position_root_model = CFRNode(game, original_player_id=game.gamestate.player_id, model=model)
            position_root_model.cfr_pred(max_iterations=40000, max_depth=40)
            position_root_model.draw_gradients("modelled.png")
            position_root_model.children[0][1].draw_gradients(name="modelled_child.png")
            print(position_root_model.cumulative_regrets)
            _, chosen_option = position_root_model.action_choice(live=True)
            print([chosen_option.name for chosen_option, _ in position_root_model.children])
            1/0

        else:
            options = game.get_options_from_state()
            chosen_option = choice(options)
            winner = chosen_option.carry_out(game)
    print(f"{winner.id} won!")
    winners[winner.id] += 1

#print(winners)
#print(winner.id)