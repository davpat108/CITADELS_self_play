from game.agent import Agent
from game.deck import Deck, Card
from game.config import building_cards, unique_building_cards, roles
import random

class RolePropery():
    def __init__(self) -> None:
        self.dead = False
        self.warrant = None
        self.possessed = False
        self.robbed = False
        self.blackmail = None
        
    def reset_at_turn_end(self):
        self.dead = False
        self.warrant = None
        self.possessed = False
        self.robbed = False
        self.blackmail = None

class Game():
    def __init__(self, avaible_roles=None, debug=False) -> None:
        if debug:
            # For debug purpuses all the unique cards are used
            used_cards = building_cards + unique_building_cards
            self.deck = Deck(used_cards=used_cards)

            # For debug purpuses 6 card per hand, 4 in general, this to test every unique building
            self.player1 = Agent(id=0)
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[0])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[0])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[0])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[1])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[3])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[4])))

            self.player2 = Agent(id=1)
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[6])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[7])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[5])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[6])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[7])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[8])))

            self.player3 = Agent(id=2)
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[10])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[12])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[9])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[10])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[11])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[12])))

            self.player4 = Agent(id=3)
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[14])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[15])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[13])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[14])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[15])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[16])))

            self.player5 = Agent(id=4)
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[15])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[0])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[17])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[18])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[19])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[20])))

            self.player6 = Agent(id=5)
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[3])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[5])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[21])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[22])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[23])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[24])))

            self.player4.crown = True

            self.roles = {0: "Assassin",
                     1: "Thief",
                     2: "Magician",
                     3: "King",
                     4: "Abbot",
                     5: "Alchemist",
                     6: "Navigator",
                     7: "Warlord"}
            

            self.turn_orders_for_roles = [0, 1, 2, 3, 4, 5]

        else:
            
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
            

        self.players = [self.player1, self.player2, self.player3, self.player4, self.player5, self.player6]
        # Dictionary with 1 item
        self.visible_face_up_role = None
        self.role_properties ={
            0 : RolePropery(),
            1 : RolePropery(),
            2 : RolePropery(),
            3 : RolePropery(),
            4 : RolePropery(),
            5 : RolePropery(),
            6 : RolePropery(),
            7 : RolePropery()
            }

        
    def sample_roles(self, avaible_roles, number_of_used_roles):
        roles = {}

        for i in range(number_of_used_roles):
            roles[i] = random.choice(avaible_roles[i])
        
        return roles