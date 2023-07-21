from game.game import Game
from itertools import combinations
from random import choice

while 1:
    game = Game()
    game.setup_round()
    winner = False
    while not winner:
        options = game.next()
        chosen_option = choice(options)
        print(chosen_option.name)
        winner = chosen_option.carry_out(game)
    print(winner.id)

