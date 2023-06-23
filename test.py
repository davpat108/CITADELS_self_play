from game.game import Game
from itertools import combinations

game = Game(debug=True)

game.player1.role = "Assassin"
game.visible_face_up_role = {1:"Thief"}
options = game.player1.character_options(game)

options[0].carry_out_assasination(game)
x = 0