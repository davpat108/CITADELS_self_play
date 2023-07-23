import unittest
from game.game import Game
from random import choice

class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = Game(debug=True)
        self.game.setup_round()
        self.sequence_of_actions = [("role_pick", "Assassin"), ("role_pick", "Spy"), ("role_pick", "Magician"), ("role_pick", "King"), ("role_pick", "Abbot"), ("role_pick", "Alchemist"), ("gold_or_card", "gold"), ("assassination", 1), ("build", )]
        self.game.roles_to_choose_from = dict(sorted(list(self.game.roles.items()), key=lambda x: x[0]))

    def test_game_setup(self):
        self.game.setup_round()
        # Add assertions here to check the initial game state

    def test_game_play(self):

        winner = False
        i=0
        while not winner:
            options = self.game.get_options_from_state()
            chosen_option = choice(options)
            for choice in options:
                if hasattr(choice, "choice"):
                    if choice.name == self.sequence_of_actions[i][0] and choice.choice == self.sequence_of_actions[i][1]:
                        chosen_option = choice
                        i+=1
                        break
                elif hasattr(choice, "target"):
                    if choice.name == self.sequence_of_actions[i][0] and choice.target == self.sequence_of_actions[i][1]:
                        chosen_option = choice
                        i+=1
                        break
                elif choice.name == "empty_option":
                    chosen_option = choice
                    break
                else:
                    raise Exception("Wrong option")
            winner = chosen_option.carry_out(self.game)
            if i == len(self.sequence_of_actions):
                asserts = True
                break
            # Add assertions here to check the game state after each action
        print(winner.id)


if __name__ == '__main__':
    unittest.main()