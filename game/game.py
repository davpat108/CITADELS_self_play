from game.agent import Agent
from game.deck import Deck, Card
from game.config import building_cards, unique_building_cards, roles, role_to_role_id
import random
from game.helper_classes import RolePropery, GameState

class Game():
    def __init__(self, avaible_roles=None, debug=False) -> None:
        if debug:
            # For debug purpuses all the unique cards are used
            used_cards = building_cards + unique_building_cards
            self.deck = Deck(used_cards=used_cards)
            self.discard_deck = Deck(empty=True)

            # For debug purpuses 6 card per hand, 4 in general, this to test every unique building
            self.player1 = Agent(id=0, playercount=6)
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[0])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[0])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[0])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[1])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[3])))
            self.player1.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[4])))

            self.player2 = Agent(id=1, playercount=6)
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[6])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[7])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[5])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[6])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[7])))
            self.player2.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[8])))

            self.player3 = Agent(id=2, playercount=6)
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[10])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[12])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[9])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[10])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[11])))
            self.player3.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[12])))

            self.player4 = Agent(id=3, playercount=6)
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[14])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[15])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[13])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[14])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[15])))
            self.player4.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[16])))

            self.player5 = Agent(id=4, playercount=6)
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[15])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[0])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[17])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[18])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[19])))
            self.player5.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[20])))

            self.player6 = Agent(id=5, playercount=6)
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[3])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[5])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[21])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[22])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[23])))
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**unique_building_cards[24])))

            self.players = [self.player1, self.player2, self.player3, self.player4, self.player5, self.player6]
            self.player4.crown = True

            self.roles = {0: "Assassin",
                     1: "Spy",
                     2: "Magician",
                     3: "King",
                     4: "Abbot",
                     5: "Alchemist",
                     6: "Navigator",
                     7: "Warlord"}
            

            self.turn_orders_for_roles = [0, 1, 2, 3, 4, 5]

        else:
            
            self.deck = Deck()
            self.discard_deck = Deck(empty=True)
            self.player1 = Agent(id=0, playercount=6)
            self.player2 = Agent(id=1, playercount=6)
            self.player3 = Agent(id=2, playercount=6)
            self.player4 = Agent(id=3, playercount=6)
            self.player5 = Agent(id=4, playercount=6)
            self.player6 = Agent(id=5, playercount=6)

            for _ in range(4):
                self.player1.hand.add_card(self.deck.draw_card())
                self.player2.hand.add_card(self.deck.draw_card())
                self.player3.hand.add_card(self.deck.draw_card())
                self.player4.hand.add_card(self.deck.draw_card())
                self.player5.hand.add_card(self.deck.draw_card())
                self.player6.hand.add_card(self.deck.draw_card())

            # All roles from config if not specified
            if avaible_roles is None:
                avaible_roles = roles
            self.roles = self.sample_roles(avaible_roles, 8)
            
            self.turn_orders_for_roles = [0, 1, 2, 3, 4, 5]
            random.shuffle(self.turn_orders_for_roles)

            crown_player = random.randint(0, 5)
            self.players = [self.player1, self.player2, self.player3, self.player4, self.player5, self.player6]
            for player in self.players:
                if player.id == crown_player:
                    player.crown = True
            
        self.seer_taken_card = []

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
        
        game_state = GameState()

        
    def sample_roles(self, avaible_roles, number_of_used_roles):
        roles = {}

        for i in range(number_of_used_roles):
            roles[i] = random.choice(avaible_roles[i])
        
        return roles
    
    def setup_round(self):
        for role_property in self.role_properties:
            role_property.reset_role_properties() 
        self.seer_taken_card = []
        
        self.used_roles = []
        # setup roles
        roles_to_choose_from = list(self.roles.items())
        random.shuffle(roles_to_choose_from)
        # Not taking into account < 4 players and 7 players
        if len(self.players) < 6:
            for _ in range(6-len(self.players)):
                self.visible_face_up_role = roles_to_choose_from.pop()
        else:
            self.visible_face_up_role = None
        # Facedown role card
        roles_to_choose_from.pop()

        self.roles_to_choose_from = dict(roles_to_choose_from.sort())
        crowned_player_index = next((player.id for player in self.players if player.crown), None)
        if crowned_player_index is None:
            raise Exception("No player with crown")
        self.turn_orders_for_roles = self.turn_orders_for_roles[crowned_player_index:] + self.turn_orders_for_roles[:crowned_player_index]

        self.gamestate.state = 0
        self.gamestate.player = self.turn_orders_for_roles[0]

        for player in self.players:
            player.substract_from_known_hand_confidences()


