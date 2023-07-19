from game.game import Game
from itertools import combinations
from random import choice

while 1:
    game = Game()
    game.setup_round()
    winner = False
    while not winner:
        options = game.next()
        #print(choice(options).name)
        winner = choice(options).carry_out(game)
    print(winner.id)

