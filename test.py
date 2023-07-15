from game.game import Game
from itertools import combinations

game = Game()
game.setup_round()
while 1:
    options = game.next()
    print(options[-1].name)
    options[-1].carry_out(game)


print("X")