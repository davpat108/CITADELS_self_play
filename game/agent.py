import random
from copy import copy
from itertools import (combinations, combinations_with_replacement,
                       permutations, product)
from game.agent_functions import reveal_warrant_as_magistrate_options ,witch_options, pick_role_options, gold_or_card_options, which_card_to_keep_options, blackmail_response_options, reveal_blackmail_as_blackmailer_options, graveyard_options, reveal_warrant_as_magistrate_options, main_round_options, seer_give_back_card, scholar_give_back_options, wizard_take_from_hand_options, emperor_options

import numpy as np
import torch.nn.functional as F
import torch

from game.config import role_to_role_id
from game.deck import Card, Deck
from game.helper_classes import GameState, HandKnowledge, RoleKnowlage
from game.option import option

class Agent():

    def __init__(self, id:int, playercount) -> None:
        # Game start
        self.hand = Deck(empty=True)
        self.role = None
        self.buildings = Deck(empty=True)

        self.just_drawn_cards = Deck(empty=True)
        self.replicas = False
        
        self.museum_cards = Deck(empty=True)
        self.can_use_lighthouse = False

        self.crown = False
        self.gold = 2
        self.id = id
        self.known_hands = []
        self.known_roles = [RoleKnowlage(player_id=i, possible_roles={}) for i in range(playercount)]
        self.first_to_7 = False
        self.witch = False


    def __eq__(self, other):
        if isinstance(other, Agent):
            return self.id == other.id and self.role == other.role and self.hand == other.hand and self.buildings == other.buildings and self.just_drawn_cards == other.just_drawn_cards and self.replicas == other.replicas and self.museum_cards == other.museum_cards and self.can_use_lighthouse == other.can_use_lighthouse and self.crown == other.crown and self.gold == other.gold and self.first_to_7 == other.first_to_7
        return False
    
    
    # gamestates:	0 - choose role -> 1
	#1 2 gold or 2 cards -> 2 or 3
	#2 Which card to put back -> 3
	#3 Blackmail response -> 4 different character
	#4 Respond to response reveal -> 5 different character
	#5 Character ability/build/smithy/museum/lab/magic_school/weapon_storage -> next player 1 or 0 or end
	#6 graveyard -> 5 different character (warlord)
    #7 Magistrate reveal -> 5 different character
    #8 seer give back card
    #9 Scholar_give put back cards
    #10 Wizard choose from hand

    #def get_weighted_output(self, game, model):
    #    encoded_game = game.encode_game(self.id)
    #    outputs = model(encoded_game)


    def get_options(self, game):
        if game.gamestate.state == 0:
            return pick_role_options(self, game)
        # If bewitched role_id is not in role properties
        if self.role == "Bewitched" or not game.role_properties[role_to_role_id[self.role]].dead:
            if game.gamestate.state == 1:
                return gold_or_card_options(self, game)
            if game.gamestate.state == 2:
                return which_card_to_keep_options(self, game)
            if game.gamestate.state == 3:
                return blackmail_response_options(self, game)
            if game.gamestate.state == 4: # interrupting
                return reveal_blackmail_as_blackmailer_options(self, game)
            if game.gamestate.state == 6: # interrupting
                return graveyard_options(self, game)
            if game.gamestate.state == 7: # interrupting
                return reveal_warrant_as_magistrate_options(self, game)
            if self.role == "Witch":
                return witch_options(self, game)
            if not game.role_properties[role_to_role_id[self.role]].possessed:
                if game.gamestate.state == 5:
                    return main_round_options(self, game)
                if game.gamestate.state == 8:
                    return seer_give_back_card(self, game)
                if game.gamestate.state == 9:
                    return scholar_give_back_options(self, game)
                if game.gamestate.state == 10:
                    return wizard_take_from_hand_options(self, game)
            else:
                return [option(name="finish_round", perpetrator=self.id, next_witch=True, crown=True if self.role == "King" or self.role == "Patrician" else False)]
        elif self.role == "Emperor" and "character_ability" not in game.gamestate.already_done_moves:
            return emperor_options(self, game, dead_emperor=True)
        else:
            return [option(name="finish_round", perpetrator=self.id, next_witch=False, crown=True if self.role == "King" or self.role == "Patrician" else False)]


    # Helper functions for agent
    def get_build_limit(self):
        if self.role == "Architect":
            build_limit = 3
        elif self.role == "Scholar":
            build_limit = 2
        elif self.role == "Bishop":
            build_limit = 0
        elif self.role == "Navigator":
            build_limit = 0
        else:
            build_limit = 1
        return build_limit

    def substract_from_known_hand_confidences_and_clear_wizard(self):
        remaining_hand_knowledge = [] # New list to keep track of remaining hand knowledge objects
        for hand_knowlage in self.known_hands:
            hand_knowlage.confidence -= 1
            hand_knowlage.wizard = False
            hand_knowlage.used = False
            if hand_knowlage.confidence != 0:
                remaining_hand_knowledge.append(hand_knowlage) # Add to new list if confidence is not 0

        self.known_hands = remaining_hand_knowledge # Assign new list to self.known_hands

    def reset_known_roles(self):
        for role_knowlage in self.known_roles:
            role_knowlage.possible_roles = {}
            role_knowlage.confirmed = False

    def count_points(self):
        points = 0
        for card in self.buildings.cards:
            points += card.cost
            if card.type_ID == 18 or card.type_ID == 23:
                points += 2
            # Whishing well
            if Card(**{"suit":"unique", "type_ID":31, "cost": 5}) in self.buildings.cards and card.suit == "unique":
                points += 1

        if len(self.buildings.cards) >= 7:
            points += 2

        if self.first_to_7:
            points += 4

        points += len(self.museum_cards.cards)

        # Imp teasury
        if Card(**{"suit":"unique", "type_ID":37, "cost": 4}) in self.buildings.cards:
            points += self.gold

        # Maproom
        if Card(**{"suit":"unique", "type_ID":39, "cost": 5}) in self.buildings.cards:
            points += len(self.hand.cards)


        return points
