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
        options = game.next()
        tota_options += len(options)
        chosen_option = choice(options)
        print(chosen_option.name)
        winner = chosen_option.carry_out(game)

    print(tota_options/i)
    print(winner.id)

