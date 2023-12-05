import random
from copy import copy
from itertools import (combinations, combinations_with_replacement,
                       permutations, product)

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
            return self.pick_role_options(game)
        # If bewitched role_id is not in role properties
        if self.role == "Bewitched" or not game.role_properties[role_to_role_id[self.role]].dead:
            if game.gamestate.state == 1:
                return self.gold_or_card_options(game)
            if game.gamestate.state == 2:
                return self.which_card_to_keep_options(game)
            if game.gamestate.state == 3:
                return self.blackmail_response_options(game)
            if game.gamestate.state == 4: # interrupting
                return self.reveal_blackmail_as_blackmailer_options(game)
            if game.gamestate.state == 6: # interrupting
                return self.graveyard_options(game)
            if game.gamestate.state == 7: # interrupting
                return self.reveal_warrant_as_magistrate_options(game)
            if self.role == "Witch":
                return self.witch_options(game)
            if not game.role_properties[role_to_role_id[self.role]].possessed:
                if game.gamestate.state == 5:
                    return self.main_round_options(game)
                if game.gamestate.state == 8:
                    return self.seer_give_back_card(game)
                if game.gamestate.state == 9:
                    return self.scholar_give_back_options(game)
                if game.gamestate.state == 10:
                    return self.wizard_take_from_hand_options(game)
            else:
                return [option(name="finish_round", perpetrator=self.id, next_witch=True, crown=True if self.role == "King" or self.role == "Patrician" else False)]
        elif self.role == "Emperor" and "character_ability" not in game.gamestate.already_done_moves:
            return self.emperor_options(game, dead_emperor=True)
        else:
            return [option(name="finish_round", perpetrator=self.id, next_witch=False, crown=True if self.role == "King" or self.role == "Patrician" else False)]

