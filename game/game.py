from game.agent import Agent
from game.deck import Deck, Card
from game.config import building_cards, unique_building_cards, roles, role_to_role_id
import random
from game.helper_classes import RolePropery, GameState
from copy import deepcopy, copy
import numpy as np

class Game():
    def __init__(self, avaible_roles=None, debug=False) -> None:
        if debug:
            # For debug purpuses all the unique cards are used
            self.used_cards = building_cards + unique_building_cards
            self.deck = Deck(used_cards=self.used_cards)
            self.used_cards = deepcopy(self.deck)
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
            self.player6.hand.add_card(self.deck.get_a_card_like_it(Card(**building_cards[0])))

            self.players = [self.player1, self.player2, self.player3, self.player4, self.player5, self.player6]
            self.player4.crown = True

            self.roles = {0: "Assassin",
                     1: "Spy",
                     2: "Wizard",
                     3: "King",
                     4: "Abbot",
                     5: "Alchemist",
                     6: "Navigator",
                     7: "Warlord"}
            

            self.turn_orders_for_roles = [0, 1, 2, 3, 4, 5]

        else:
            self.used_cards = building_cards + random.sample(unique_building_cards, 14)
            self.deck = Deck(used_cards=self.used_cards)
            self.used_cards = deepcopy(self.deck)
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
        
        self.gamestate = GameState()
        self.ending = False
        self.terminal = False

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Game):
            return self.players == __value.players and self.deck == __value.deck and self.discard_deck == __value.discard_deck and self.used_cards == __value.used_cards and self.gamestate == __value.gamestate and self.ending == __value.ending and self.terminal == __value.terminal
        return False
        
    def sample_roles(self, avaible_roles, number_of_used_roles):
        roles = {}

        for i in range(number_of_used_roles):
            roles[i] = random.choice(avaible_roles[i])
        
        return roles
    
    def setup_round(self):
        for role_property in self.role_properties.values():
            role_property.reset_role_properties() 
        
        # Choosen roles during chose role phase, filled at the end of rolepick
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
        

        self.roles_to_choose_from = dict(sorted(roles_to_choose_from, key=lambda x: x[0]))
        crowned_player_index = next((player.id for player in self.players if player.crown), None)
        self.turn_orders_for_roles = self.turn_orders_for_roles[crowned_player_index:] + self.turn_orders_for_roles[:crowned_player_index]

        self.gamestate = GameState(state=0, player_id=self.turn_orders_for_roles[0])

        for player in self.players:
            player.substract_from_known_hand_confidences_and_clear_wizard()
            player.reset_known_roles()

    def is_last_round(self, player:Agent):
        "Returns whether its the games last round or not"
        if not self.ending:
            for player in self.players:
                if len(player.buildings.cards) == 7:
                    self.ending = True
                    player.first_to_7 = True

    def get_unknown_cards(self, player_character):
        unknown_cards = deepcopy(self.used_cards) # Start with a copy of all used cards
        unknown_cards.cards = unknown_cards.cards*3

        # Remove cards that any player has in their buildings deck
        for p in self.players:
            for card in p.buildings.cards:
                unknown_cards.get_a_card_like_it(card)

        # Remove cards that any player has in their museum_cards deck
        for p in self.players:
            for card in p.museum_cards.cards:
                unknown_cards.get_a_card_like_it(card)

        # Remove cards that the player has in their hand
        for card in player_character.hand.cards:
            unknown_cards.get_a_card_like_it(card)

        # Remove cards that are in the HandKnowledge of the player_character
        for hk in player_character.known_hands:
            if hk.used: # Only remove cards if HandKnowledge is used
                # If no such card in unknown_cards,
                # all of those cards are in the player's hand, built or in museum
                # (Its faster to do this instead of picking them one by one when they are revealed)
                try:
                    for card in hk.hand.cards:
                        unknown_cards.get_a_card_like_it(card)
                except KeyError:
                    hk.hand.get_a_card_like_it(card)

        return unknown_cards

    def sample_private_information(self, player_character):
        # Decide whether to use each HandKnowledge based on confidence, (confidence * 20% chance)
        for hk in player_character.known_hands:
            random_chance = random.random()
            if (hk.confidence - 1) * 0.2 > random_chance:
                hk.used = False
            else:
                hk.used = False

        unknown_cards = self.get_unknown_cards(player_character)
        known_roles_by_player = deepcopy(player_character.known_roles)

        # Settle deck
        deck_count = len(self.deck.cards)
        lighthouse_knowledge = deepcopy(next((hk for hk in player_character.known_hands if hk.player_id == -1 and hk.used), None))
        self.deck.cards = []
        if lighthouse_knowledge is not None:
            # Number of cards to take from HandKnowledge
            known_card_count = min(len(lighthouse_knowledge.hand.cards), deck_count)

            for _ in range(known_card_count):
                card_to_take = lighthouse_knowledge.hand.cards.pop(0)
                unknown_cards.get_a_card_like_it(card_to_take)
                self.deck.add_card(card_to_take)
                deck_count -= 1

        # Fill the deck with unknown cards if more than lighthouse
        random.shuffle(unknown_cards.cards)
        for _ in range(deck_count):
            self.deck.add_card(unknown_cards.draw_card())

        # Settle players
        for player in self.players:
            if player == player_character:
                continue

            # Check if player exists in HandKnowledge list and if it is used
            player_hand_knowledge = deepcopy(next((hk for hk in player_character.known_hands if hk.player_id == player.id and hk.used), None))

            player_hand_card_count = len(player.hand.cards)
            player.hand.cards = []
            if player_hand_knowledge is not None:
                # Number of cards to take from HandKnowledge
                known_card_count = min(len(player_hand_knowledge.hand.cards), player_hand_card_count)
                for _ in range(known_card_count):
                    card_to_take = player_hand_knowledge.hand.cards.pop(0)
                    unknown_cards.get_a_card_like_it(card_to_take)
                    player.hand.add_card(card_to_take)
                    player_hand_card_count -= 1

            # Fill the rest of the hand with random cards
            for _ in range(player_hand_card_count):
                player.hand.add_card(unknown_cards.draw_card())
            
            # Sample blackmails
            blackmails = [key for key, role_property in self.role_properties.items() if role_property.blackmail is not None]
            if blackmails:
                random.shuffle(blackmails)
                real_blackmail_key = blackmails[0]
                for key in blackmails:
                    self.role_properties[key].blackmail = "Real" if key == real_blackmail_key else "Fake"
            
            # Sample warrants
            warrants = [key for key, role_property in self.role_properties.items() if role_property.warrant is not None]
            if warrants:
                random.shuffle(warrants)
                real_warrant_key = warrants[0]
                for key in warrants:
                    self.role_properties[key].warrant = "Real" if key == real_warrant_key else "Fake"

            # Sample roles
            if self.gamestate.state != 0:
                player.role = random.choice(list(known_roles_by_player[player.id].possible_roles.values()))

                # Remove the chosen role from all RoleKnowledge objects
                for rk in known_roles_by_player:
                    rk.possible_roles = {k: v for k, v in rk.possible_roles.items() if v != player.role}

        if self.gamestate.state != 0:
            self.refresh_used_roles()
        
            if not self.gamestate.interruption:
                self.setup_next_player(current_player_id=self.gamestate.player_id)

        # Replace the game's deck with the remaining unknown cards
        self.deck.cards = unknown_cards.cards

    def refresh_used_roles(self):
        # Clear the used_roles list
        self.used_roles = []

        # Iterate over all players and add their role ID to used_roles
        for player in self.players:
            role_id = role_to_role_id[player.role]
            self.used_roles.append(role_id)
        self.used_roles.sort()

    def check_game_ending(self):
        if self.ending:
            points = []
            for player in self.players:
                points.append(player.count_points())
            self.points = points
            self.terminal = True
            self.rewards = np.zeros(len(self.players))
            self.rewards[points.index(max(points))] = 1
            return self.players[points.index(max(points))]
        return False


    def setup_next_player(self, current_player_id=None):
        if self.gamestate.state == 0:
            self.refresh_used_roles()
            self.gamestate.state = 1
            self.gamestate.player_id = self.get_player_from_role_id(self.used_roles[0]).id
        elif current_player_id is not None:
            self.gamestate.state = 1
            self.gamestate.player_id = self.get_player_from_role_id(self.used_roles[self.used_roles.index(role_to_role_id[self.players[current_player_id].role])+1]).id
            self.gamestate.already_done_moves = []
        else:
            raise Exception("No current player and not in rolepick state")

    def get_player_from_role_id(self, role_id):
        for player in self.players:
            if player.role == self.roles[role_id]:
                return player
        return None


    def get_options_from_state(self):
        # returns the next actor.get_options() for the next player
        return self.players[self.gamestate.player_id].get_options(self)