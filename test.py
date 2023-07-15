from game.game import Game
from itertools import combinations

game = Game()
game.setup_round()
while 1:
    options = game.next()
    print(options[0].name)
    options[0].carry_out(game)


print("X")