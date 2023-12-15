from copy import deepcopy, copy
from game.deck import Deck, Card
from game.helper_classes import GameState, HandKnowledge, RolePropery, RoleKnowlage
from game.config import role_to_role_id
import numpy as np
import torch

class option():
    def __init__(self, name, **kwargs):
        self.name = name
        self.attributes = kwargs
        self.generate_name_to_id_map()

    def __eq__(self, other):
        return self.name == other.name and self.attributes == other.attributes
    
    def __str__(self):
        return f"{self.name}, {self.attributes}"

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

    # others
    def generate_name_to_id_map(self):
        actions = [
            "role_pick", "gold_or_card", "which_card_to_keep", "blackmail_response",
            "reveal_blackmail_as_blackmailer", "reveal_warrant_as_magistrate", "build", "empty_option",
            "finish_round", "ghost_town_color_choice", "smithy_choice", "laboratory_choice",
            "magic_school_choice", "weapon_storage_choice", "lighthouse_choice", "museum_choice",
            "graveyard", "take_gold_for_war", "assassination", "magistrate_warrant", "bewitching",
            "steal", "blackmail", "spy", "magic_hand_change", "discard_and_draw", "look_at_hand",
            "take_from_hand", "seer", "give_back_card", "take_crown_king", "give_crown",
            "take_crown_pat", "bishop", "cardinal_exchange", "abbot_gold_or_card", "abbot_beg",
            "merchant", "alchemist", "trader", "architect", "navigator_gold_card", "scholar",
            "scholar_card_pick", "warlord_desctruction", "marshal_steal", "diplomat_exchange"
        ]

        # Create a dictionary that maps each action to a unique ID
        self.name_to_id = {action: idx for idx, action in enumerate(actions)}



    def encode_option(self):
        """
        bits
        47 what option 0-47
        6 who 47-53
        6 whom 53-60
        8 what 60-68
        8 what fake 68-76
        13 named 76-89
        40 card 89-129
        1 replica 129-130
        1 gold or card 130-131
        """
        encoded_option = torch.zeros([1, 131])
        encoded_option[0, self.name_to_id[self.name]] = 1
        encoded_option[0, self.attributes['perpetrator'] + 47] = 1

        names = {
            "gold": 0,
            "card": 1,
            "pay": 2,
            "not_pay": 3,
            "reveal": 4,
            "not_reveal": 5,
            "4gold" :6,
            "4card": 7,
            "trade": 8,
            "war": 9,
            "religion": 10,
            "lord":11,
            "unique": 12,
        }

        if "target" in self.attributes.keys():# 6 different
            encoded_option[0, self.attributes['target'] + 53] = 1
        elif "choice" in self.attributes.keys() and isinstance(self.attributes['choice'], str) and self.attributes["choice"] in role_to_role_id.keys():#8 different
            encoded_option[0, role_to_role_id[self.attributes['choice']] + 60] = 1
        elif "choice" in self.attributes.keys() and isinstance(self.attributes['choice'], str):# 13 different
            encoded_option[0, names[self.attributes['choice']] + 76] = 1
        elif "choice" in self.attributes.keys() and isinstance(self.attributes['choice'], Card): # 40 different
            encoded_option[0, self.attributes['choice'].type_ID + 89] = 1
        elif "choice" in self.attributes.keys() and isinstance(self.attributes['choice'], list): # the same 40 multiple values possible even 2s or more
            for card in self.attributes['choice']:
                encoded_option[0, card.type_ID + 89] = 1
        elif "built_card" in self.attributes.keys():# the same 40
            encoded_option[0, self.attributes['built_card'].type_ID + 89] = 1
        elif "real_target" in self.attributes.keys(): # 8 same as the first
            encoded_option[0, self.attributes['real_target'] + 60] = 1
        elif "fake_targets" in self.attributes.keys(): # 8 different two ones
            encoded_option[0, self.attributes['fake_targets'][0] + 68] = 1
            encoded_option[0, self.attributes['fake_targets'][1] + 68] = 1
        elif "fake_target" in self.attributes.keys():#8 same as the secound but only one
            encoded_option[0, self.attributes['fake_target'] + 68] = 1
        elif "replica" in self.attributes.keys():# different one bit 1 or 0
            encoded_option[0, 129] = self.attributes['replica']
        elif "gold_or_card_combination" in self.attributes.keys():# different one bit same counting
            encoded_option[0, 130] = self.attributes['gold_or_card_combination'].count("card")
        elif "chosen_card" in self.attributes.keys():# the same 40
            encoded_option[0, self.attributes['chosen_card'].type_ID + 89] = 1
        elif "cards_to_give" in self.attributes.keys():# the same 40 multiple values possible even 2s or more
            for card in self.attributes['cards_to_give']:
                encoded_option[0, card.type_ID + 89] = card.type_ID 

        return encoded_option


    def carry_out(self, game):
        winner = game.action_map[self.name](self, game)
        game.is_last_round(game.players[self.attributes['perpetrator']])

        return winner

