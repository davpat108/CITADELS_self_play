from game.game import Game
from itertools import combinations

#ame = Game(debug=False)
#
#ef empty():
#   return
#
#rint([1,2,3]+empty())

combs = combinations([1,2], 10)

for comb in combs:
    print(comb)