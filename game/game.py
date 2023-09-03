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
            self.roles = self.sample_roles_for_player(avaible_roles, 8)
            
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
        
    def encode_avalible_roles(self, config_roles):
        # Create a mapping of roles to their index
        role_to_index = {}
        for key, role_list in config_roles.items():
            for idx, role in enumerate(role_list):
                role_to_index[role] = idx

        # Create an 8x3 numpy array filled with zeros
        encoded_array = np.zeros((8, 3), dtype=int)

        # Iterate over the self.roles dictionary and fill in the one-hot encoding
        for key, role in self.roles.items():
            encoded_array[key][role_to_index[role]] = 1

        return encoded_array
    
    def encode_player_roles(self):
        # Create a 6x8 numpy array filled with zeros
        encoded_array = np.zeros((6, 8), dtype=int)

        # Iterate over the players and fill in the one-hot encoding
        for player in self.players:
            if player.role is None or player.role == "Bewitched":
                continue
            role_index = role_to_role_id.get(player.role)
            if role_index is not None:
                encoded_array[player.id][role_index] = 1

        return encoded_array


    def encode_role_properties(role_properties):
        """
        Encodes the role properties into an 8x5 numpy array.
        The function checks each attribute of the RoleProperty object in the order:
        'dead', 'warrant', 'possessed', 'robbed', 'blackmail'. If the attribute is 
        not None or not False, the corresponding position in the numpy array is set to 1.

        Args:
        - role_properties (dict): A dictionary where keys are role IDs and values are RoleProperty objects.

        Returns:
        - numpy.ndarray: An 8x5 numpy array representing the encoded role properties.
        """

        encoded_array = np.zeros((8, 5), dtype=int)
        attributes_order = ['dead', 'warrant', 'possessed', 'robbed', 'blackmail']

        for role_id, role_property in role_properties.items():
            for idx, attribute in enumerate(attributes_order):
                value = getattr(role_property, attribute)
                if value:
                    encoded_array[role_id][idx] = 1

        return encoded_array

    def encode_output(self, player_id):
        encoded_avalible_roles = self.encode_avalible_roles(roles)
        player_roles= self.encode_player_roles()
        encoded_player_hand, encoded_hand_suits = self.players[player_id].hand.encode_deck()

        encoded_built_cards, encoded_buildings_suits = zip(*[player.buildings.encode_deck() for player in self.players])
        encoded_built_cards = np.vstack(encoded_built_cards)
        encoded_buildings_suits = np.vstack(encoded_buildings_suits)

        encoded_player_ID = np.zeros(6, dtype=int)
        encoded_player_ID[player_id] = 1

        encoded_just_drawn_cards, encoded_just_drawn_suits = self.players[player_id].just_drawn_cards.encode_deck()

        encoded_role_properties = self.encode_role_properties()

        encoded_array = np.concatenate([
        encoded_avalible_roles.flatten(),
        player_roles.flatten(),
        encoded_player_hand.flatten(),
        encoded_hand_suits.flatten(),
        encoded_built_cards.flatten(),
        encoded_buildings_suits.flatten(),
        encoded_player_ID.flatten(),
        encoded_just_drawn_cards.flatten(),
        encoded_just_drawn_suits.flatten(),
        encoded_role_properties.flatten()
        ])

        return encoded_array


    def sample_roles_for_player(self, avaible_roles, number_of_used_roles):
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
        unknown_cards.cards = unknown_cards.cards

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

    def sample_private_information(self, player_character, role_sample = True):
        print("Sampling private information")
        # Decide whether to use each HandKnowledge based on confidence, (confidence * 20% chance)
        for hk in player_character.known_hands:
            random_chance = random.random()
            if (hk.confidence - 1) * 0.2 > random_chance:
                hk.used = True
            else:
                hk.used = False

        unknown_cards = self.get_unknown_cards(player_character)

        self.sample_deck(player_character, unknown_cards)
        self.sample_warrants_and_blackmails()
        # Settle players
        known_roles_by_player = deepcopy(player_character.known_roles)
        if role_sample:
            self.remove_role_and_smaller_id_roles_from_role_knowledge_if_unconfirmed(self.players[self.gamestate.player_id].role, known_roles_by_player)
        for player in self.players:
            self.sample_cards_for_opponent(player, player_character, unknown_cards)
            if role_sample:
                self.sample_roles_for_opponent(player, player_character, known_roles_by_player)

        if role_sample:
            self.refresh_roles_after_sampling_roles()


        # Replace the game's deck with the remaining unknown cards

        self.check_if_all_cards_exist()


    def sample_deck(self, player_character, unknown_cards):
        # Settle deck
        deck_count = len(self.deck.cards)
        lighthouse_knowledge = deepcopy(next((hk for hk in player_character.known_hands if hk.player_id == -1 and hk.used), None))
        self.deck.cards = []
        if lighthouse_knowledge is not None:
            # Number of cards to take from HandKnowledge
            known_card_count = min(len(lighthouse_knowledge.hand.cards), deck_count)

            for _ in range(known_card_count):
                card_to_take = lighthouse_knowledge.hand.cards.pop(0)
                self.deck.add_card(card_to_take)
                deck_count -= 1

        # Fill the deck with unknown cards if more than lighthouse
        random.shuffle(unknown_cards.cards)
        for _ in range(deck_count):
            self.deck.add_card(unknown_cards.draw_card())

    def sample_cards_for_opponent(self, player, original_player, unknown_cards):
        if player == original_player:
            return
        # Check if player exists in HandKnowledge list and if it is used
        player_hand_knowledge = deepcopy(next((hk for hk in original_player.known_hands if hk.player_id == player.id and hk.used), None))
        player_hand_card_count = len(player.hand.cards)
        player.hand.cards = []
        if player_hand_knowledge is not None:
            # Number of cards to take from HandKnowledge
            known_card_count = min(len(player_hand_knowledge.hand.cards), player_hand_card_count)
            for _ in range(known_card_count):
                card_to_take = player_hand_knowledge.hand.cards.pop(0)
                player.hand.add_card(card_to_take)
                player_hand_card_count -= 1
        # Fill the rest of the hand with random cards
        for _ in range(player_hand_card_count):
            player.hand.add_card(unknown_cards.draw_card())


    def sample_roles_for_opponent(self, player, original_player, known_roles_by_player, role_pick_end_sample=False):
        # Dont sample for original player or the current playing beginning its round
        if original_player == player or (player.id == self.gamestate.player_id and not role_pick_end_sample):
            return
        if self.gamestate.state != 0:
            player.role = random.choice(list(known_roles_by_player[player.id].possible_roles.values()))
            # Remove the chosen role from all RoleKnowledge objects
            self.remove_role_from_role_knowledge(player.role, known_roles_by_player)


    def remove_role_from_role_knowledge(self, role, role_knowledge_list):
        for rk in role_knowledge_list:
            if not rk.confirmed:
                rk.possible_roles = {k: v for k, v in rk.possible_roles.items() if v != role}
            

    def remove_role_and_smaller_id_roles_from_role_knowledge_if_unconfirmed(self, role, known_roles_by_player):
        # Remove the players role whose about to play from the samples, and all the roles that are smaller id
        self.remove_role_from_role_knowledge(role, known_roles_by_player)
        if role is not None:
            logically_left_out_role_ids = [role_id for role_id in self.roles.keys() if role_id < role_to_role_id[role]]
            for role_id in logically_left_out_role_ids:
                self.remove_role_from_role_knowledge(self.roles[role_id], known_roles_by_player)

    # UNUSED
    def sample_private_info_after_role_pick_end(self, original_player):
        print("Sampling private information after role pick end")
        known_roles_by_player = deepcopy(original_player.known_roles)
        for player in self.players:
            self.sample_roles_for_opponent(player, original_player, known_roles_by_player, role_pick_end_sample=True)
        self.refresh_roles_after_sampling_roles(from_role_pick_end_sample=True)

    # UNUSED
    def is_end_of_role_pick(self):
        # If someone is bewitched, its not the end of role pick, and witch is also changed
        if "Bewitched" in [player.role for player in self.players]:
            return False
        return self.gamestate.state == 1 and self.gamestate.player_id == self.get_player_from_role_id(self.used_roles[0]).id

    def sample_warrants_and_blackmails(self):
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


    def refresh_roles_after_sampling_roles(self, from_role_pick_end_sample=False):
        # refresh orders after sampling roles
        if self.gamestate.state != 0:
            if not from_role_pick_end_sample: # It is already done later in setup next player
                self.refresh_used_roles()
        
            if not self.gamestate.interruption and from_role_pick_end_sample:
                self.setup_next_player(current_player_id=self.gamestate.player_id, from_role_pick_end_sample=from_role_pick_end_sample)


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

    def check_if_all_cards_exist(self):
        total_cards = []
        for player in self.players:
            total_cards += player.hand.cards
            total_cards += player.buildings.cards
            total_cards += player.museum_cards.cards
            total_cards += player.just_drawn_cards.cards
        total_cards += self.deck.cards
        total_cards += self.discard_deck.cards
        
        used_card_ids = [card.type_ID for card in self.used_cards.cards]
        total_card_ids = [card.type_ID for card in total_cards]

        missing_from_used = [card_id for card_id in total_card_ids if card_id not in used_card_ids]
        missing_from_total = [card_id for card_id in used_card_ids if card_id not in total_card_ids]

        if missing_from_used or missing_from_total:
            print("Missing cards")
            #raise Exception(f"Card error in sampling. Missing from used cards: {missing_from_used}. Missing from total cards: {missing_from_total}.")

    def setup_next_player(self, current_player_id=None, from_role_pick_end_sample=False):
        if self.gamestate.state == 0 or from_role_pick_end_sample:
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
        if role_id == -1:
            for player in self.players:
                if player.role == "Bewitched":
                    return player
        else:
            for player in self.players:
                if player.role == self.roles[role_id]:
                    return player
        return None


    def get_options_from_state(self):
        # returns the next actor.get_options() for the next player
        return self.players[self.gamestate.player_id].get_options(self)