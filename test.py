from game.game import Game
from itertools import combinations
from random import choice

while 1:
    game = Game()
    game.setup_round()
    winner = False
    i =0
    while not winner:
        i+=1
        options = game.next()
        chosen_option = choice(options)
        print(chosen_option.name)
        winner = chosen_option.carry_out(game)
        if i > 100:
            #game.sample_private_information(game.player1)
            print("X")
    print(winner.id)

