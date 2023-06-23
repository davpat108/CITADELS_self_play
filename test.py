from game.game import Game
from itertools import combinations

game = Game(debug=True)

#game.player1.role = "Assassin"
#game.visible_face_up_role = {1:"Thief"}
#options = game.player1.character_options(game)
#
#options[0].carry_out_assasination(game)
#x = 0

game.player1.role = "Spy"
game.visible_face_up_role = {0:"Assassin"}
options = game.player1.character_options(game)

options[0].carry_out_spying(game)
x = 0