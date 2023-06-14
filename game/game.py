from agent import Agent
from deck import Deck
from config import building_cards, unique_building_cards, roles
import random
class Game():
    def __init__(self, avaible_roles=None, debug=False) -> None:
        if debug:
            # For debug purpuses all the unique cards are used
            used_cards = building_cards + unique_building_cards
            self.deck = Deck(used_cards=used_cards)

            # For debug purpuses 6 card per hand, 4 in general, this to test every unique building
            self.player1 = Agent(id=0)
            self.player1.hand.add_card(self.deck.get_a_card_like_it(0))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(0))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(16))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(17))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(18))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(19))

            self.player2 = Agent(id=1)
            self.player2.hand.add_card(self.deck.get_a_card_like_it(6))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(7))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(20))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(21))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(22))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(23))

            self.player3 = Agent(id=2)
            self.player3.hand.add_card(self.deck.get_a_card_like_it(10))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(12))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(24))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(25))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(26))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(27))

            self.player4 = Agent(id=3)
            self.player4.hand.add_card(self.deck.get_a_card_like_it(14))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(15))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(28))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(29))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(30))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(31))

            self.player5 = Agent(id=4)
            self.player5.hand.add_card(self.deck.get_a_card_like_it(15))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(15))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(32))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(33))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(34))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(35))

            self.player6 = Agent(id=5)
            self.player6.hand.add_card(self.deck.get_a_card_like_it(4))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(5))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(36))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(37))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(38))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(39))

            self.player4.crown = True

            self.roles = {0: ["Witch"],
                     1: ["Thief"],
                     2: ["Magician"],
                     3: ["King"],
                     4: ["Abbot"],
                     5: ["Alchemist"],
                     6: ["Navigator"],
                     7: ["Warlord"]}
            
            self.turn_orders_for_roles = [0, 1, 2, 3, 4, 5]

        elif avaible_roles is not None:
            self.deck = Deck()
            self.player1 = Agent(id=0)
            self.player2 = Agent(id=1)
            self.player3 = Agent(id=2)
            self.player4 = Agent(id=3)
            self.player5 = Agent(id=4)
            self.player6 = Agent(id=5)

            for _ in range(4):
                self.player1.hand.add_card(self.deck.draw_card())
                self.player2.hand.add_card(self.deck.draw_card())
                self.player3.hand.add_card(self.deck.draw_card())
                self.player4.hand.add_card(self.deck.draw_card())
                self.player5.hand.add_card(self.deck.draw_card())
                self.player6.hand.add_card(self.deck.draw_card())

            # All roles if not specified
            if avaible_roles is None:
                avaible_roles = roles
            self.roles = self.sample_roles(avaible_roles, 8)
            
            self.turn_orders_for_roles = [0, 1, 2, 3, 4, 5]
            random.shuffle(self.turn_orders_for_roles)

            crown_player = random.randint(0, 5)

            for player in [self.player1, self.player2, self.player3, self.player4, self.player5, self.player6]:
                if player.id == crown_player:
                    player.crown = True



    def sample_roles(self, avaible_roles, number_of_used_roles):
        roles = {}

        for i in range(number_of_used_roles):
            roles[i] = random.choice(avaible_roles[i])
        
        return roles