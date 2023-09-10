from game.game import Game
from itertools import combinations
from random import choice

while 1:
    game = Game()
    game.setup_round()
    winner = False
    i =0
    tota_options = 0
    while not winner:
        i+=1
        options, _ = game.get_options_from_state()
        options = [option for option_list in options.values() for option in option_list]
        tota_options += len(options)
        chosen_option = choice(options)
        print(chosen_option.name)
        winner = chosen_option.carry_out(game)
        x = game.encode_game()

    print(tota_options/i)
    print(winner.id)

