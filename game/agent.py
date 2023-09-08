import random
from copy import copy
from itertools import combinations, combinations_with_replacement, permutations

import numpy as np
import torch.nn.functional as F

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
                return [option(name="finish_round", perpetrator=self.id, next_witch=True, crown=True if self.role == "King" or self.role == "Patrician" else False)], self.create_mask(game.game_model_output_size, 0, 0)
        elif self.role == "Emperor" and "character_ability" not in game.gamestate.already_done_moves:
            return self.emperor_options(game, dead_emperor=True)
        else:
            return [option(name="finish_round", perpetrator=self.id, next_witch=False, crown=True if self.role == "King" or self.role == "Patrician" else False)], self.create_mask(game.game_model_output_size, 0, 0)

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

    # Options from gamestates
    def pick_role_options(self, game):
        return [option(name="role_pick", perpetrator=self.id, choice=role) for role in game.roles_to_choose_from.values()], [self.create_mask(game.game_model_output_size, 6, 14, list(game.roles_to_choose_from.keys()))]

    def gold_or_card_options(self, game):
        return [option(name="gold_or_card", perpetrator=self.id, choice="gold"), option(name="gold_or_card", perpetrator=self.id, choice="card")] if len(game.deck.cards) > 1 else [option(name="gold_or_card", perpetrator=self.id, choice="gold")], [self.create_mask(game.game_model_output_size, 14, 15)]
    
    def which_card_to_keep_options(self, game):
        options = []
        if Card(**{"suit":"unique", "type_ID":20, "cost": 4}) in self.buildings.cards:
            card_choices = list(combinations(self.just_drawn_cards.cards, 2))
            for card_choice in card_choices:
                options.append(option(name="which_card_to_keep", perpetrator=self.id, choice=card_choice))
            return options
        return [option(name="which_card_to_keep", perpetrator=self.id, choice=[card]) for card in self.just_drawn_cards.cards], [self.create_mask(game.game_model_output_size, 15, 54, [card.type_ID for card in self.just_drawn_cards.cards])]

    def blackmail_response_options(self, game) -> list:
        if game.role_properties[role_to_role_id[self.role]].blackmail:
            return [option(choice="pay", perpetrator=self.id, name="blackmail_response"), option(choice="not_pay", perpetrator=self.id, name="blackmail_response")], [self.create_mask(game.game_model_output_size, 54, 55)]
        return [option(name="empty_option", perpetrator=self.id, next_gamestate=GameState(state=5, player_id=self.id))], [self.create_mask(game.game_model_output_size, 54, 55)]
    
    def reveal_blackmail_as_blackmailer_options(self, game) -> list:
        return [option(choice="reveal", perpetrator=self.id, target=game.gamestate.next_gamestate.player_id, name="reveal_blackmail_as_blackmailer"), option(choice="not_reveal", perpetrator=self.id, target=game.gamestate.next_gamestate.player_id, name="reveal_blackmail_as_blackmailer")], [self.create_mask(game.game_model_output_size, 55, 56)]
    
    def reveal_warrant_as_magistrate_options(self, game) -> list:
        return [option(choice="reveal", perpetrator=self.id, target=game.gamestate.next_gamestate.player_id, name="reveal_warrant_as_magistrate"), option(choice="not_reveal", perpetrator=self.id, target=game.gamestate.next_gamestate.player_id, name="reveal_warrant_as_magistrate")], [self.create_mask(game.game_model_output_size, 56, 57)]


    def ghost_town_color_choice_options(self) -> list:
        # Async
        if Card(**{"suit":"unique", "type_ID":19, "cost": 2}) in self.buildings.cards:
            return [option(choice="trade", perpetrator=self.id, name="ghost_town_color_choice"), option(choice="war", perpetrator=self.id, name="ghost_town_color_choice"),
                     option(choice="religion", perpetrator=self.id, name="ghost_town_color_choice"), option(choice="lord", perpetrator=self.id, name="ghost_town_color_choice"), option(choice="unique", perpetrator=self.id, name="ghost_town_color_choice")]
        return []
        
    def smithy_options(self, game) -> list:
        if Card(**{"suit":"unique", "type_ID":21, "cost": 5}) in self.buildings.cards and self.gold >= 2 and not "smithy" in game.gamestate.already_done_moves:
            return [option(name="smithy_choice", perpetrator=self.id)], self.create_mask(game.game_model_output_size, 1397, 1398)
        return [], self.create_mask(game.game_model_output_size, 1397, 1397)
    
    def laboratory_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":22, "cost": 5}) in self.buildings.cards and not "lab" in game.gamestate.already_done_moves:
            for card in self.hand.cards:
                options.append(option(name="laboratory_choice", perpetrator=self.id, choice=card)), self.create_mask(game.game_model_output_size, 1398, 1399)
        return [], self.create_mask(game.game_model_output_size, 1399, 1399)
    
    def magic_school_options(self, game) -> list:
        # Every round
        # used before character ability
        if not "magic_school" in game.gamestate.already_done_moves:
            if Card(**{"suit":"unique", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"trade", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"war", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"religion", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"lord", "type_ID":25, "cost": 6}) in self.buildings.cards:
                return [option(choice="trade", perpetrator=self.id, name="magic_school_choice"), option(choice="war", perpetrator=self.id, name="magic_school_choice"),
                         option(choice="religion", perpetrator=self.id, name="magic_school_choice"), option(choice="lord", perpetrator=self.id, name="magic_school_choice"), option(choice="unique", perpetrator=self.id, name="magic_school_choice")], [self.create_mask(game.game_model_output_size, 1399, 1400), self.create_mask(game.game_model_output_size, 1400, 1406)]
        return [], [self.create_mask(game.game_model_output_size, 1406, 1406), self.create_mask(game.game_model_output_size, 1406, 1406)]
    
    def weapon_storage_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":27, "cost": 3}) in self.buildings.cards:
            for player in game.players:
                if player.id != self.id:
                    for card in player.buildings.cards:
                        options.append(option(perpetrator=self.id,target=player.id, choice = card, name="weapon_storage_choice"))
        used_bits = [] if options == [] else [0]
        return options, [self.create_mask(game.game_model_output_size, 1406, 1407, used_bits)]
    
    def lighthouse_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":29, "cost": 3}) in self.buildings.cards and self.can_use_lighthouse:
            for card in game.deck.cards:
                options.append(option(choice=card, perpetrator=self.id, name="lighthouse_choice"))
        used_bits = [] if options == [] else [card.type_ID for card in game.deck.cards]
        return options, [self.create_mask(game.game_model_output_size, 1407, 1408), self.create_mask(game.game_model_output_size, 1408, 1447, used_bits)]

    def museum_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":34, "cost": 4}) in self.buildings.cards and not "museum" in game.gamestate.already_done_moves:
            for card in self.hand.cards:
                options.append(option(choice=card, perpetrator=self.id, name="museum_choice"))
        used_bits = [] if options == [] else [card.type_ID for card in self.hand.cards]
        return options, [self.create_mask(game.game_model_output_size, 1447, 1448), self.create_mask(game.game_model_output_size, 1448, 1487, used_bits)]

    # Main stuff
    def get_builds(self, options) -> list:
        # Returns buildable cards from hand by cost
        for card in self.hand.cards:
            cost = card.cost
            replica = 0
            if Card(**{"suit":"unique", "type_ID":35, "cost": 6}) in self.buildings.cards and card.suit == "unique":
                cost -= -1
            if card in self.buildings.cards and Card(**{"suit":"unique", "type_ID":36, "cost": 5}) and not self.replicas: 
                replica = self.replicas + 1
            if cost <= self.gold and option(name="build", perpetrator=self.id, built_card=card, replica=replica) not in options:
                options.append(option(name="build", perpetrator=self.id, built_card=card, replica=replica))


    def build_options(self, game):
        options = []
        build_limit = self.get_build_limit()
        if self.role != "Trader":
            if game.gamestate.already_done_moves.count("trade_building") + game.gamestate.already_done_moves.count("non_trade_building") < build_limit:
                self.get_builds(options)
        else:
            if game.gamestate.already_done_moves.count("non_trade_building") < build_limit:
                self.get_builds(options)
        build_possibe_bit = [0] if option.name == "build" else []
        return options, [self.create_mask(game.game_model_output_size, 58, 59, build_possibe_bit), self.create_mask(game.game_model_output_size, 59, 98, list(set([option.built_card.type_ID for option in options])))]  # 0 is for the first bit, whether to build or not


    def main_round_options(self, game):
        options = [option(name="finish_round", perpetrator=self.id, next_witch=False, crown=False)]

        build_options, build_mask = self.build_options(game)
        character_options, char_mask = self.character_options(game)
        smithy_options, smithy_mask = self.smithy_options(game)
        lab_options, lab_mask = self.laboratory_options(game)
        rocks_options, rocks_mask = self.magic_school_options(game)
        ws_options, ws_mask = self.weapon_storage_options(game)
        lh_options, lh_mask = self.lighthouse_options(game)
        museum_options, museum_mask = self.museum_options(game)

        options += build_options + character_options + smithy_options + lab_options + rocks_options + ws_options + lh_options + museum_options
        all_masks = [build_mask, char_mask, smithy_mask, lab_mask, rocks_mask, ws_mask, lh_mask, museum_mask]
        combined_mask = np.logical_and.reduce(all_masks).astype(int)
        return options, combined_mask
    
    
    def graveyard_options(self, game):
        if self.gold > 0:
            return [option(name="graveyard", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 57, 58)]
        return [option(name="empty_option", perpetrator=self.id, next_gamestate=game.gamestate.next_gamestate)], [self.create_mask(game.game_model_output_size, 57, 57)]


    def character_options(self, game):
        all_options = []
        all_masks = []

        # ID 0
        if "character_ability" not in game.gamestate.already_done_moves:
            role_functions = {
                "Assassin": self.assasin_options,
                "Magistrate": self.magistrate_options,

                "Thief": self.thief_options,
                "Blackmailer": self.blackmail_options,
                "Spy": self.spy_options,

                "Magician": self.magician_options,
                "Wizard": self.wizard_look_at_hand_options,
                "Seer": self.seer_options,

                "King": self.king_options,
                "Emperor": self.emperor_options,
                "Patrician": self.patrician_options,

                "Bishop": self.bishop_options,
                "Cardinal": self.cardinal_options,
                "Abbot": self.abbot_options,

                "Merchant": self.merchant_options,
                "Alchemist": self.alchemist_options,
                "Trader": self.trader_options,

                "Architect": self.architect_options,
                "Navigator": self.navigator_options,
                "Scholar": self.scholar_options,

                "Warlord": self.warlord_options,
                "Marshal": self.marshal_options,
                "Diplomat": self.diplomat_options


            }

            if self.role in role_functions:
                options, masks = role_functions[self.role](game)
                all_options += options
                all_masks.append(masks)

        elif self.role in ["Warlord", "Marshal", "Diplomat"]:
            options, masks = self.take_gold_for_war_options(game)
            all_options += options
            all_masks.append(masks)
        elif self.role == "Abbot":
            options, masks = self.abbot_beg(game)
            all_options += options
            all_masks.append(masks)
        # ID 8
        # Not yet

        # Compute the logical AND of all masks
        primary_mask = [mask[0] for mask in all_masks]
        secoundary_mask = [mask[1] for mask in all_masks if len(mask)>1]
        tetriary_mask = [mask[2] for mask in all_masks if len(mask)>2]
        cardinal_mask = [mask[3] for mask in all_masks if len(mask)>3]

        combined_primary_mask = np.logical_and.reduce(primary_mask).astype(int)
        combined_secoundary_mask = np.logical_and.reduce(secoundary_mask).astype(int) if secoundary_mask != [] else None
        combined_tetriary_mask = np.logical_and.reduce(tetriary_mask).astype(int) if tetriary_mask != [] else None
        combined_cardinal_mask = np.logical_and.reduce(cardinal_mask).astype(int) if cardinal_mask != [] else None

        return all_options, [combined_primary_mask, combined_secoundary_mask, combined_tetriary_mask, combined_cardinal_mask]



    # ID 0
    def assasin_options(self, game):
        target_possibilities = game.roles.copy()
        if game.visible_face_up_role:
            target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
        used_bits = list(range(8))
        used_bits.remove(0) #remove the asssasin
        return [option(name="assassination", perpetrator=self.id, target=role_ID) for role_ID in target_possibilities.keys() if role_ID > 0], [self.create_mask(game.game_model_output_size, 99, 100), self.create_mask(game.game_model_output_size, 100, 108, used_bits)]
        
        
    def magistrate_options(self, game):
        options = []
        # [1:] to remove the magistrate itself
        target_possibilities = list(game.roles.keys())[1:]
        # Remove the visible left out role
        if game.visible_face_up_role:
            target_possibilities.remove(next(iter(game.visible_face_up_role.keys())))

        for real_target in target_possibilities:
            for fake_tagets in combinations(target_possibilities, 2):
                if real_target not in fake_tagets:
                    options.append(option(name="magistrate_warrant", perpetrator=self.id, real_target=real_target, fake_targets=list(fake_tagets)))
        used_bits = list(range(8))
        used_bits.remove(0) #remove the magistrate
        return options, [self.create_mask(game.game_model_output_size, 108, 109), self.create_mask(game.game_model_output_size, 109, 117, used_bits), self.create_mask(game.game_model_output_size, 117, 125, used_bits)]

    def witch_options(self, game):
        target_possibilities = game.roles.copy()
        if game.visible_face_up_role:
            target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
        used_bits = list(range(8))
        used_bits.remove(0) #remove the witch
        return [option(name="bewitching",  perpetrator=self.id, target=role_ID) for role_ID in target_possibilities.keys() if role_ID > 0], [self.create_mask(game.game_model_output_size, 125, 126), self.create_mask(game.game_model_output_size, 126, 134, used_bits)]
    
    
    # ID 1
    def thief_options(self, game):
        target_possibilities = game.roles.copy()
        if game.visible_face_up_role:
            target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
        used_bits = list(range(8))
        used_bits.remove(1) #remove the thief
        return [option(name="steal", perpetrator=self.id, target=role_ID) for role_ID in target_possibilities.keys() if role_ID > 1], [self.create_mask(game.game_model_output_size, 134, 135), self.create_mask(game.game_model_output_size, 135, 143, used_bits)]
    
    def blackmail_options(self, game):
        options = []
        # [2:] to remove the blackmailer and the ID 0 character (assassin and the like)
        target_possibilities = list(game.roles.keys())[2:]
        # Remove the visible face up role
        if game.visible_face_up_role:
            target_possibilities.remove(next(iter(game.visible_face_up_role.keys())))
        
        # Cant blackmail possessed role
        for role_property in game.role_properties.items():
            if role_property[1].possessed:
                target_possibilities.remove(role_property[0])

        for targets in combinations(target_possibilities, 2):
            options.append(option(name="blackmail", perpetrator=self.id, real_target=targets[0], fake_target=targets[1]))
            options.append(option(name="blackmail", perpetrator=self.id, real_target=targets[1], fake_target=targets[0]))
        used_bits = list(range(8))
        used_bits.remove(1) #remove the magistrate
        return options, [self.create_mask(game.game_model_output_size, 143, 144), self.create_mask(game.game_model_output_size, 144, 152, used_bits), self.create_mask(game.game_model_output_size, 152, 160, used_bits)]
    
    def spy_options(self, game):
        options = []
        for player in game.players:
            if player.id != self.id:
                for suit in ["trade", "war", "religion", "lord", "unique"]:
                    options.append(option(name="spy", perpetrator=self.id, target=player.id, suit=suit))
        used_bits = list(range(6))
        used_bits.remove(self.player_id)
        return options, [self.create_mask(game.game_model_output_size, 160, 161), self.create_mask(game.game_model_output_size, 161, 167, used_bits)]
    
    # ID 2
    def magician_options(self, game):
        options = []
        for player in game.players:
            if player.id != self.id:
                options.append(option(name="magic_hand_change", perpetrator=self.id, target=player.id))
                
        for r in range(1, len(self.hand.cards)+1):
            # list of cards
            discard_possibilities = list(combinations(self.hand.cards, r))
            for i in range(0, len(discard_possibilities), max(round(len(discard_possibilities)/1e2), 1)):
                options.append(option(name="discard_and_draw", perpetrator=self.id, cards=discard_possibilities[i]))

        used_bits_person_target = list(range(6))
        used_bits_person_target.remove(self.player_i)
        used_bits_which_card = list(set([card.type_ID for card in self.hand.cards]))
        return options, [self.create_mask(game.game_model_output_size, 167, 169), self.create_mask(game.game_model_output_size, 169, 175, used_bits_person_target), self.create_mask(game.game_model_output_size, 175, 215, used_bits_which_card)]

    def wizard_look_at_hand_options(self, game):
        options = []
        for player in game.players:
            if player.id != self.id and len(player.hand.cards) > 0:
                options.append(option(name="look_at_hand", perpetrator=self.id, target=player.id))
        used_bits = list(range(6))
        used_bits.remove(self.player_id)
        return options, [self.create_mask(game.game_model_output_size, 215, 216, used_bits), self.create_mask(game.game_model_output_size, 216, 222, used_bits)]
    
    def wizard_take_from_hand_options(self, game):
        target_hand = next((hand for hand in self.known_hands if hand.confidence == 5 and hand.player_id != -1 and hand.wizard==True), None)
        options = []
        replica = 0
        for card in target_hand.hand.cards:
            if option(name="take_from_hand", card=card, build=False, perpetrator=self.id, target=target_hand.player_id) not in options:
                options.append(option(name="take_from_hand", card=card, build=False, perpetrator=self.id, target=target_hand.player_id))
            cost = card.cost
            if Card(**{"suit":"unique", "type_ID":35, "cost": 6}) in self.buildings.cards and card.suit == "unique":
                cost -= -1
            if card in self.buildings.cards: 
                replica = self.replicas + 1
            if cost <= self.gold and option(name="take_from_hand", built_card=card, build=True, perpetrator=self.id, target=target_hand.player_id, replica=replica) not in options:
                options.append(option(name="take_from_hand", built_card=card, build=True, perpetrator=self.id, target=target_hand.player_id, replica=replica))
        if options == []:
            return [option(name="empty_option", perpetrator=self.id, next_gamestate=game.gamestate.next_gamestate)]
        used_bits = list(set([option.card.type_ID if not option.build else option.built_card.type_ID + 39 for option in options]))
        return options, [self.create_mask(game.game_model_output_size, 222, 300, used_bits)]

    def seer_options(self, game):
        return [option(name="seer", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 300, 301)]


    def seer_give_back_card(self, game):
        def random_permutations(cards, k, position=0, num_permutations=3):
            """
            For each card in cards, place it in the specified position and generate num_permutations random orderings
            from the remaining cards for the other positions.
            """
            all_perms = []

            for card in cards:
                
                remaining_cards = [c for c in cards if c != card]

                for _ in range(num_permutations):
                    random.shuffle(remaining_cards)
                    perm_with_position = list(remaining_cards[:k-1])  # Take the first k-1 cards after shuffling
                    perm_with_position.insert(position, card)
                    all_perms.append(tuple(perm_with_position))

            return all_perms

        options = []
        k = len(game.seer_taken_card_from)

        for pos in range(k):
            perms = random_permutations(self.hand.cards, k, position=pos)
            for perm in perms:
                card_handouts = {player_card_pair[0]: player_card_pair[1] for player_card_pair in zip(game.seer_taken_card_from, perm)}
                options.append(option(name="give_back_card", perpetrator=self.id, card_handouts=card_handouts))

        return options, [self.create_mask(game.game_model_output_size, 301, 535, [card.type_ID + i * 39 for i in range(6) for card in self.hand.cards])]

    # ID 3
    def king_options(self, game):
        # Nothing you just take the crown
        return [option(name="take_crown_king", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 535, 536)]

    def emperor_options(self, game, dead_emperor=False):
        options = []
        for player in game.players:
            if player.id != self.id:
                if len(player.hand.cards) and not dead_emperor:
                    options.append(option(name="give_crown", perpetrator=self.id, target=player.id, gold_or_card="card"))
                if player.gold and not dead_emperor:
                    options.append(option(name="give_crown", perpetrator=self.id, target=player.id, gold_or_card="gold"))
                if not player.gold and not len(player.hand.cards) or dead_emperor:
                    options.append(option(name="give_crown", perpetrator=self.id,  target=player.id, gold_or_card="nothing"))
        used_bits = list(range(18))
        used_bits.remove(self.player_id)
        used_bits.remove(self.player_id+6)
        used_bits.remove(self.player_id+12)
        return options, [self.create_mask(game.game_model_output_size, 536, 537, used_bits), self.create_mask(game.game_model_output_size, 537, 555, used_bits)]
    
    def patrician_options(self, game):
        # Nothing you just take the crown
        return [option(name="take_crown_pat", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 555, 556)]

    # ID 4
    def bishop_options(self, game):
        # Nothing you just can't be warlorded
        return [option(name="bishop", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 556, 557)]

    def cardinal_options(self, game):
        options = []
        for player in game.players:
            # Check each card in our hand
            for card in self.hand.cards:
                # If the card cost is less than the player's gold
                cost = card.cost
                factory = False
                replica = 0
                if Card(**{"suit":"unique", "type_ID":35, "cost": 6}) in self.buildings.cards and card.suit == "unique":
                    cost -= -1
                    factory = True
                if card in self.buildings.cards and Card(**{"suit":"unique", "type_ID":36, "cost": 5}) and not self.replicas: 
                    replica = self.replicas+1
                if cost <= player.gold:
                    # Calculate how many cards we need to give in exchange
                    exchange_cards_count = max(player.gold - cost, 0)
                    # If we have enough cards to give (excluding the current card)
                    if len(self.hand.cards) - 1 >= exchange_cards_count:
                        # Get all combinations of exchange_cards_count cards (excluding the current card)
                        other_cards = [c for c in self.hand.cards if c != card]
                        exchange_combinations = list(combinations(other_cards, exchange_cards_count))
                        # Each combination of exchange cards is a possible trade
                        for i in range(0, len(exchange_combinations), max(round(len(exchange_combinations)/1e2), 1)):
                            options.append(option(name="cardinal_exchange", perpetrator=self.id, target=player.id, built_card=card, cards_to_give=exchange_combinations[i], replica=replica, factory=factory))
        used_bits_whom = list(range(6))
        used_bits_whom.remove(self.player_id)
        used_bits_build = list(set([option.built_card.card_ID for option in options]))
        used_bits_give = list(set([card.card_ID for card in self.hand.cards]))

        return options, [self.create_mask(game.game_model_output_size, 557, 558), self.create_mask(game.game_model_output_size, 558, 564, used_bits_whom)]#, self.create_mask(game.game_model_output_size, 564, 603, used_bits_build), self.create_mask(game.game_model_output_size, 603, 642, used_bits_give)]
    
    def abbot_options(self, game):
        options = []
        total_religious_districts = sum([1 if card.suit == "religion" else 0 for card in self.hand.cards])
        gold_or_card_combinations = combinations_with_replacement(["gold", "card"], total_religious_districts)
        for gold_or_card_combination in gold_or_card_combinations:
            options.append(option(name="abbot_gold_or_card", perpetrator=self.id,  gold_or_card_combination=list(gold_or_card_combination)))
        return options, [self.create_mask(game.game_model_output_size, 642, 643), self.create_mask(game.game_model_output_size, 643, 645)]
    
    def abbot_beg(self, game):
        if not "begged" in game.gamestate.already_done_moves:
            return [option(name="abbot_beg", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 645, 646)]
        return [], [self.create_mask(game.game_model_output_size, 645, 645)]
    
    #ID 5
    def merchant_options(self, game):
        # Nothing to choose
        return [option(name="merchant", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 646, 647)]
    
    def alchemist_options(self, game):
        # Nothing to choose
        return [], [self.create_mask(game.game_model_output_size, 647, 647)]
    
    def trader_options(self, game):
        # Nothing to choose
        return [option(name="trader", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 647, 648)]
    
    # ID 6
    def architect_options(self, game):
        return [option(name="architect", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 648, 649)]
    
    def navigator_options(self, game):
        return [option(name="navigator_gold_card", perpetrator=self.id, choice="4gold"), option(name="navigator_gold_card", perpetrator=self.id, choice="4card")], [self.create_mask(game.game_model_output_size, 649, 650), self.create_mask(game.game_model_output_size, 650, 651)]

    def scholar_options(self, game):
        if game.deck.cards:
            return [option(name="scholar", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 651, 652)]
        return [], [self.create_mask(game.game_model_output_size, 651, 651)]

    def scholar_give_back_options(self, game):
        options = []
        seven_drawn_cards = game.seven_drawn_cards
        for card in seven_drawn_cards.cards:
            unchosen_cards = copy(seven_drawn_cards)
            unchosen_cards.get_a_card_like_it(card)
            options.append(option(name="scholar_card_pick", choice=card, perpetrator=self.id,  unchosen_cards=unchosen_cards))
        used_bits = list(set([option.choice.type_ID for option in options]))
        return options, [self.create_mask(game.game_model_output_size, 652, 691, used_bits)]

    # ID 7
    def warlord_options(self, game):
        options = []
        for player in game.players:
            if len(player.buildings.cards) < 7:
                for building in player.buildings.cards:
                    if building.cost-1 <= self.gold and building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop":
                        if option(name="warlord_desctruction", target=player.id, perpetrator=self.id, choice=building) not in options:
                            options.append(option(name="warlord_desctruction", target=player.id, perpetrator=self.id, choice=building))
        used_bits = [card.type_ID * (player.id + 1) for player in game.players if player.id != self.id for card in player.buildings]
        return options, [self.create_mask(game.game_model_output_size, 692, 693, used_bits), self.create_mask(game.game_model_output_size, 693, 927, used_bits)]

    def marshal_options(self, game):
        options = []
        for player in game.players:
            if len(player.buildings.cards) < 7 and player.id != self.id:
                for building in player.buildings.cards:
                    if building.cost <= self.gold and building.cost <= 3 and building not in self.buildings.cards and building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop":
                        if option(name="marshal_steal", target=player.id, perpetrator=self.id, choice=building) not in options:
                            options.append(option(name="marshal_steal", target=player.id, perpetrator=self.id, choice=building))
        used_bits = [card.type_ID * (player.id + 1) for player in game.players if player.id != self.id for card in player.buildings]
        return options, [self.create_mask(game.game_model_output_size, 927, 928, used_bits), self.create_mask(game.game_model_output_size, 928, 1162, used_bits)]

    def diplomat_options(self, game):
        options = []
        for player in game.players:
            if len(player.buildings.cards) < 7 and player.id != self.id:
                for enemy_building in player.buildings.cards:
                    for own_building in self.buildings.cards:
                        if enemy_building.cost-own_building.cost <= self.gold and enemy_building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop" and enemy_building not in self.buildings.cards:
                            if option(name="diplomat_exchange", target=player.id, perpetrator=self.id, choice=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)) not in options:
                                options.append(option(name="diplomat_exchange", target=player.id, perpetrator=self.id, choice=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)))
        used_bits = [card.type_ID * (player.id + 1) for player in game.players if player.id != self.id for card in player.buildings]
        return options, [self.create_mask(game.game_model_output_size, 1162, 1163, used_bits), self.create_mask(game.game_model_output_size, 1163, 1397, used_bits)]
    
    def take_gold_for_war_options(self, game):
        if "take_gold" not in game.gamestate.already_done_moves:
            return [option(name="take_gold_for_war", perpetrator=self.id)], [self.create_mask(game.game_model_output_size, 691, 692)]
        return [], [self.create_mask(game.game_model_output_size, 691, 691)]
    # ID 8 later for more players


    def create_mask(self, model_output_size, start_index, end_index, pick_indexes=None):
        """
        Create a mask vector based on the given pick indexes within a specified interval.

        Args:
        - flattened_output (numpy.ndarray): Output size of the model
        - start_index (int): The starting index of the interval.
        - end_index (int): The ending index of the interval.
        - pick_indexes (list, optional): List of indexes to pick within the interval.

        Returns:
        - numpy.ndarray: A mask vector of the same length as the flattened output, with ones at the specified pick indexes within the interval and zeros elsewhere.
        """
        mask = np.zeros(model_output_size, dtype=int)
        if pick_indexes is None:
            mask[start_index:end_index] = 1
        else:
            mask[start_index:end_index][pick_indexes] = 1
        return mask
    


    def get_distribution(self, model_output, model_masks, options):
        """
        Compute a conditional probability distribution for a set of options based on a hierarchical decision-making process.

        The function uses the top-level masks to get a probability distribution for the top-level decisions. For each 
        top-level decision, if there are subsequent decisions, it uses the corresponding masks to get a probability 
        distribution conditioned on the top-level decision. The final output is a single vector representing the 
        conditional probability distribution for all options.

        Parameters:
        - model_output (torch.Tensor): The output of the model, a vector of size (1487 currently).
        - model_masks (list of lists of torch.Tensor): A hierarchical list of masks. Each top-level list corresponds to 
          a top-level decision, and each subsequent list corresponds to subsequent decisions conditioned on the top-level 
          decision.
        - options (dict): A dictionary where keys correspond to top-level decisions and values are lists of options 
          available for each top-level decision.

        Returns:
        - list: A probability distribution over all the options, taking into account the hierarchical decision-making process.
        - list: all the options in the same order as the probability distribution

        Example:
        model_output = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7])
        model_bits = [
            [torch.tensor([True, True, False, False, False, False, False, False]),
             torch.tensor([False, False, True, True, False, False, False, False])],
            [torch.tensor([False, False, False, False, True, True, False, False])],
            [torch.tensor([False, False, False, False, False, False, True, True])]
        ]
        options = {0: ["option1", "option2"], 1: ["option3"], 2: ["option4", "option5"]}
        distribution = get_distribution(model_output, model_bits, options)
        """
        final_probs = []
    
        # Iterate over the top-level masks and corresponding options
        for i, masks in enumerate(model_masks):
            # Extract relevant bits for the top-level decision
            top_level_bits = model_masks[i][0]
            top_level_probs = F.softmax(model_output[top_level_bits], dim=0)
    
            # If there are subsequent decisions for this top-level decision
            if len(masks) > 1 and i in options:
                for j, option in enumerate(options[i]):
                    # Extract relevant bits for the subsequent decision
                    subsequent_bits = model_masks[i][1]
                    subsequent_probs = F.softmax(model_output[subsequent_bits], dim=0)
    
                    # Multiply the top-level probability with the subsequent probabilities
                    combined_probs = top_level_probs[j] * subsequent_probs
                    final_probs.extend(combined_probs.tolist())
    
            # If there are no subsequent decisions, just use the top-level probabilities
            else:
                final_probs.extend(top_level_probs.tolist())

        return final_probs, [option for option_list in options.values() for option in option_list]
