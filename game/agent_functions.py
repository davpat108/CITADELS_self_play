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

# Helper functions for agent
def get_build_limit(agent):
    if agent.role == "Architect":
        build_limit = 3
    elif agent.role == "Scholar":
        build_limit = 2
    elif agent.role == "Bishop":
        build_limit = 0
    elif agent.role == "Navigator":
        build_limit = 0
    else:
        build_limit = 1
    return build_limit
    
def substract_from_known_hand_confidences_and_clear_wizard(agent):
    remaining_hand_knowledge = [] # New list to keep track of remaining hand knowledge objects
    for hand_knowlage in agent.known_hands:
        hand_knowlage.confidence -= 1
        hand_knowlage.wizard = False
        hand_knowlage.used = False
        if hand_knowlage.confidence != 0:
            remaining_hand_knowledge.append(hand_knowlage) # Add to new list if confidence is not 0

    agent.known_hands = remaining_hand_knowledge # Assign new list to agent.known_hands

def reset_known_roles(agent):
    for role_knowlage in agent.known_roles:
        role_knowlage.possible_roles = {}
        role_knowlage.confirmed = False

def count_points(agent):
    points = 0
    for card in agent.buildings.cards:
        points += card.cost
        if card.type_ID == 18 or card.type_ID == 23:
            points += 2
        # Whishing well
        if Card(**{"suit":"unique", "type_ID":31, "cost": 5}) in agent.buildings.cards and card.suit == "unique":
            points += 1

    if len(agent.buildings.cards) >= 7:
        points += 2

    if agent.first_to_7:
        points += 4

    points += len(agent.museum_cards.cards)

    # Imp teasury
    if Card(**{"suit":"unique", "type_ID":37, "cost": 4}) in agent.buildings.cards:
        points += agent.gold
        
    # Maproom
    if Card(**{"suit":"unique", "type_ID":39, "cost": 5}) in agent.buildings.cards:
        points += len(agent.hand.cards)


    return points

# Options from gamestates
def pick_role_options(agent, game):
    return [option(name="role_pick", perpetrator=agent.id, choice=role) for role in game.roles_to_choose_from.values()]

def gold_or_card_options(agent, game):
    return [option(name="gold_or_card", perpetrator=agent.id, choice="gold"), option(name="gold_or_card", perpetrator=agent.id, choice="card")]if len(game.deck.cards) > 1 else [option(name="gold_or_card", perpetrator=agent.id, choice="gold")] 
    
def which_card_to_keep_options(agent, game):
    options = []
    if Card(**{"suit":"unique", "type_ID":20, "cost": 4}) in agent.buildings.cards:
        card_choices = list(combinations(agent.just_drawn_cards.cards, 2))
        for card_choice in card_choices:
            options.append(option(name="which_card_to_keep", perpetrator=agent.id, choice=card_choice))
        return options

    options = []
    seen_types = set()
    for card in agent.just_drawn_cards.cards:
        if card.type_ID not in seen_types:
            options.append(option(name="which_card_to_keep", perpetrator=agent.id, choice=[card]))
            seen_types.add(card.type_ID)
    return options

def blackmail_response_options(agent, game) -> list:
    if game.role_properties[role_to_role_id[agent.role]].blackmail:
        return [option(choice="pay", perpetrator=agent.id, name="blackmail_response"), option(choice="not_pay", perpetrator=agent.id, name="blackmail_response")]
    return [option(name="empty_option", perpetrator=agent.id, next_gamestate=GameState(state=5, player_id=agent.id))]
    
def reveal_blackmail_as_blackmailer_options(agent, game) -> list:
    return [option(choice="reveal", perpetrator=agent.id, target=game.gamestate.next_gamestate.player_id, name="reveal_blackmail_as_blackmailer"), option(choice="not_reveal", perpetrator=agent.id, target=game.gamestate.next_gamestate.player_id, name="reveal_blackmail_as_blackmailer")]
    
def reveal_warrant_as_magistrate_options(agent, game) -> list:
    return [option(choice="reveal", perpetrator=agent.id, target=game.gamestate.next_gamestate.player_id, name="reveal_warrant_as_magistrate"), option(choice="not_reveal", perpetrator=agent.id, target=game.gamestate.next_gamestate.player_id, name="reveal_warrant_as_magistrate")]


def ghost_town_color_choice_options(agent) -> list:
    # Async
    if Card(**{"suit":"unique", "type_ID":19, "cost": 2}) in agent.buildings.cards:
        return [option(choice="trade", perpetrator=agent.id, name="ghost_town_color_choice"), option(choice="war", perpetrator=agent.id, name="ghost_town_color_choice"),
                 option(choice="religion", perpetrator=agent.id, name="ghost_town_color_choice"), option(choice="lord", perpetrator=agent.id, name="ghost_town_color_choice"), option(choice="unique", perpetrator=agent.id, name="ghost_town_color_choice")]
    return []
        
def smithy_options(agent, game) -> list:
    if Card(**{"suit":"unique", "type_ID":21, "cost": 5}) in agent.buildings.cards and agent.gold >= 2 and not "smithy" in game.gamestate.already_done_moves:
        return [option(name="smithy_choice", perpetrator=agent.id)]
    return []
    
def laboratory_options(agent, game) -> list:
    options = []
    if Card(**{"suit":"unique", "type_ID":22, "cost": 5}) in agent.buildings.cards and not "lab" in game.gamestate.already_done_moves:
        for card in agent.hand.cards:
            options.append(option(name="laboratory_choice", perpetrator=agent.id, choice=card))
        return options
    return []
    
def magic_school_options(agent, game) -> list:
    # Every round
    # used before character ability
    if not "magic_school" in game.gamestate.already_done_moves:
        if Card(**{"suit":"unique", "type_ID":25, "cost": 6}) in agent.buildings.cards or Card(**{"suit":"trade", "type_ID":25, "cost": 6}) in agent.buildings.cards or Card(**{"suit":"war", "type_ID":25, "cost": 6}) in agent.buildings.cards or Card(**{"suit":"religion", "type_ID":25, "cost": 6}) in agent.buildings.cards or Card(**{"suit":"lord", "type_ID":25, "cost": 6}) in agent.buildings.cards:
            return [option(choice="trade", perpetrator=agent.id, name="magic_school_choice"), option(choice="war", perpetrator=agent.id, name="magic_school_choice"),
                     option(choice="religion", perpetrator=agent.id, name="magic_school_choice"), option(choice="lord", perpetrator=agent.id, name="magic_school_choice"), option(choice="unique", perpetrator=agent.id, name="magic_school_choice")]
    return []
    
def weapon_storage_options(agent, game) -> list:
    options = []
    if Card(**{"suit":"unique", "type_ID":27, "cost": 3}) in agent.buildings.cards:
        for player in game.players:
            if player.id != agent.id:
                for card in player.buildings.cards:
                    options.append(option(perpetrator=agent.id,target=player.id, choice=card, name="weapon_storage_choice"))
    return options
    
def lighthouse_options(agent, game) -> list:
    options = []
    if Card(**{"suit":"unique", "type_ID":29, "cost": 3}) in agent.buildings.cards and agent.can_use_lighthouse:
        seen_types = set()
        for card in game.deck.cards:
            if card.type_ID not in seen_types:
                options.append(option(choice=card, perpetrator=agent.id, name="lighthouse_choice"))
                seen_types.add(card.type_ID)
                
    return options

def museum_options(agent, game) -> list:
    options = []
    if Card(**{"suit":"unique", "type_ID":34, "cost": 4}) in agent.buildings.cards and not "museum" in game.gamestate.already_done_moves:
        seen_types = set()
        for card in agent.hand.cards:
            if card.type_ID not in seen_types:
                options.append(option(choice=card, perpetrator=agent.id, name="museum_choice"))
                seen_types.add(card.type_ID)
                
    return options

# Main stuff
def get_builds(agent, options):
    # Returns buildable cards from hand by cost
    for card in agent.hand.cards:
        cost = card.cost
        replica = 0
        if Card(**{"suit":"unique", "type_ID":35, "cost": 6}) in agent.buildings.cards and card.suit == "unique":
            cost -= -1
        if card in agent.buildings.cards and Card(**{"suit":"unique", "type_ID":36, "cost": 5}) and not agent.replicas: 
            replica = agent.replicas + 1
        if cost <= agent.gold and option(name="build", perpetrator=agent.id, built_card=card, replica=replica) not in options:
            options.append(option(name="build", perpetrator=agent.id, built_card=card, replica=replica))


def build_options(agent, game):
    options = []
    build_limit = agent.get_build_limit()
    if agent.role != "Trader":
        if game.gamestate.already_done_moves.count("trade_building") + game.gamestate.already_done_moves.count("non_trade_building") < build_limit:
            agent.get_builds(options)
    else:
        if game.gamestate.already_done_moves.count("non_trade_building") < build_limit:
            agent.get_builds(options)
    return options


def main_round_options(agent, game):
    
    build_options = agent.build_options(game) # 0
    character_options = agent.character_options(game) # 1-2
    smithy_options = agent.smithy_options(game) # 3
    lab_options = agent.laboratory_options(game) # 4
    rocks_options = agent.magic_school_options(game) # 5
    ws_options = agent.weapon_storage_options(game) # 6
    lh_options = agent.lighthouse_options(game) # 7
    museum_options = agent.museum_options(game) # 8
    
    finish_round_option = [option(name="finish_round", perpetrator=agent.id, next_witch=False, crown=False)]
    
    options = build_options + character_options + smithy_options + lab_options + rocks_options + ws_options + lh_options + museum_options + finish_round_option
    return options
    
    
def graveyard_options(agent, game):
    if agent.gold > 0:
        return [option(name="graveyard", perpetrator=agent.id)]
    return [option(name="empty_option", perpetrator=agent.id, next_gamestate=game.gamestate.next_gamestate)]


def character_options(agent, game):
    decisions = []

    # ID 0
    if "character_ability" not in game.gamestate.already_done_moves:
        role_functions = {
            "Assassin": agent.assasin_options,
            "Magistrate": agent.magistrate_options,

            "Thief": agent.thief_options,
            "Blackmailer": agent.blackmail_options,
            "Spy": agent.spy_options,

            "Magician": agent.magician_options,
            "Wizard": agent.wizard_look_at_hand_options,
            "Seer": agent.seer_options,

            "King": agent.king_options,
            "Emperor": agent.emperor_options,
            "Patrician": agent.patrician_options,

            "Bishop": agent.bishop_options,
            "Cardinal": agent.cardinal_options,
            "Abbot": agent.abbot_options,

            "Merchant": agent.merchant_options,
            "Alchemist": agent.alchemist_options,
            "Trader": agent.trader_options,

            "Architect": agent.architect_options,
            "Navigator": agent.navigator_options,
            "Scholar": agent.scholar_options,

            "Warlord": agent.warlord_options,
            "Marshal": agent.marshal_options,
            "Diplomat": agent.diplomat_options
        }
            
        if agent.role in role_functions:
            options = role_functions[agent.role](game)
            decisions = options


    # Handle Abbot's additional options
    if agent.role == "Abbot":
        options = agent.abbot_beg(game)
        decisions += options

    # Handle Warlord, Diplomat, and Marshal's additional options
    if agent.role in ["Warlord", "Marshal", "Diplomat"]:
        options = agent.take_gold_for_war_options(game)
        decisions += options

    return decisions


# ID 0
def assasin_options(agent, game):
    target_possibilities = game.roles.copy()
    if game.visible_face_up_role:
        target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
    
    return [option(name="assassination", perpetrator=agent.id, choice=role_ID) for role_ID in target_possibilities.keys() if role_ID > 0]
        
        
def magistrate_options(agent, game):
    options = []
    # [1:] to remove the magistrate itagent
    target_possibilities = list(game.roles.keys())[1:]
    # Remove the visible left out role
    if game.visible_face_up_role:
        target_possibilities.remove(next(iter(game.visible_face_up_role.keys())))

    for real_target in target_possibilities:
        for fake_tagets in combinations(target_possibilities, 2):
            if real_target not in fake_tagets:
                options.append(option(name="magistrate_warrant", perpetrator=agent.id, real_target=real_target, fake_targets=list(fake_tagets)))
        
    return options

def witch_options(agent, game):
    target_possibilities = game.roles.copy()
    if game.visible_face_up_role:
        target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
    used_bits = list(range(8))
    used_bits.remove(0) #remove the witch
    return [option(name="bewitching",  perpetrator=agent.id, choice=role_ID) for role_ID in target_possibilities.keys() if role_ID > 0]
    
    
# ID 1
def thief_options(agent, game):
    target_possibilities = game.roles.copy()
    if game.visible_face_up_role:
        target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
    used_bits = list(range(8))
    used_bits.remove(0)
    used_bits.remove(1) #remove the thief
    return [option(name="steal", perpetrator=agent.id, choice=role_ID) for role_ID in target_possibilities.keys() if role_ID > 1]
    
def blackmail_options(agent, game):
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
        options.append(option(name="blackmail", perpetrator=agent.id, real_target=targets[0], fake_target=targets[1]))
        options.append(option(name="blackmail", perpetrator=agent.id, real_target=targets[1], fake_target=targets[0]))

    return options

def spy_options(agent, game):
    options = []
    for player in game.players:
        if player.id != agent.id:
            for suit in ["trade", "war", "religion", "lord", "unique"]:
                options.append(option(name="spy", perpetrator=agent.id, target=player.id, suit=suit))
    
    return options
 
# ID 2
def magician_options(agent, game):
    options = []
    for player in game.players:
        if player.id != agent.id:
            options.append(option(name="magic_hand_change", perpetrator=agent.id, target=player.id))
                
    for r in range(1, len(agent.hand.cards)+1):
        # list of cards
        discard_possibilities = list(combinations(agent.hand.cards, r))
        for i in range(0, len(discard_possibilities), max(round(len(discard_possibilities)/1e2), 1)):
            options.append(option(name="discard_and_draw", perpetrator=agent.id, cards=discard_possibilities[i]))

    return options

def wizard_look_at_hand_options(agent, game):
    options = []
    used_bits = list(range(6))
    used_bits.remove(agent.id)
    for player in game.players:
        if player.id != agent.id and len(player.hand.cards) > 0:
            options.append(option(name="look_at_hand", perpetrator=agent.id, target=player.id))
        if player.id != agent.id and len(player.hand.cards) == 0:
            used_bits.remove(player.id)

    return options

def wizard_take_from_hand_options(agent, game):
    target_hand = next((hand for hand in agent.known_hands if hand.confidence == 5 and hand.player_id != -1 and hand.wizard==True), None)
    options = []
    replica = 0
    for card in target_hand.hand.cards:
        if option(name="take_from_hand", card=card, build=False, perpetrator=agent.id, target=target_hand.player_id) not in options:
            options.append(option(name="take_from_hand", card=card, build=False, perpetrator=agent.id, target=target_hand.player_id))
        cost = card.cost
        if Card(**{"suit":"unique", "type_ID":35, "cost": 6}) in agent.buildings.cards and card.suit == "unique":
            cost -= -1
        if card in agent.buildings.cards: 
            replica = agent.replicas + 1
        if cost <= agent.gold and option(name="take_from_hand", built_card=card, build=True, perpetrator=agent.id, target=target_hand.player_id, replica=replica) not in options:
            options.append(option(name="take_from_hand", built_card=card, build=True, perpetrator=agent.id, target=target_hand.player_id, replica=replica))
    if options == []:
        return [option(name="empty_option", perpetrator=agent.id, next_gamestate=game.gamestate.next_gamestate)]
    return options

def seer_options(agent, game):
    return [option(name="seer", perpetrator=agent.id)]


def seer_give_back_card(agent, game):
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
        perms = random_permutations(agent.hand.cards, k, position=pos)
        for perm in perms:
            card_handouts = {player_card_pair[0]: player_card_pair[1] for player_card_pair in zip(game.seer_taken_card_from, perm)}
            options.append(option(name="give_back_card", perpetrator=agent.id, card_handouts=card_handouts))

    return options

# ID 3
def king_options(agent, game):
    # Nothing you just take the crown
    return [option(name="take_crown_king", perpetrator=agent.id)]

def emperor_options(agent, game, dead_emperor=False):
    options = []
    for player in game.players:
        if player.id != agent.id:
            if len(player.hand.cards) and not dead_emperor:
                options.append(option(name="give_crown", perpetrator=agent.id, target=player.id, gold_or_card="card"))
            if player.gold and not dead_emperor:
                options.append(option(name="give_crown", perpetrator=agent.id, target=player.id, gold_or_card="gold"))
            if not player.gold and not len(player.hand.cards) or dead_emperor:
                options.append(option(name="give_crown", perpetrator=agent.id,  target=player.id, gold_or_card="nothing"))
    used_bits = list(range(18))
    used_bits.remove(agent.id)
    used_bits.remove(agent.id+6)
    used_bits.remove(agent.id+12)
    return options

def patrician_options(agent, game):
    # Nothing you just take the crown
    return [option(name="take_crown_pat", perpetrator=agent.id)]

# ID 4
def bishop_options(agent, game):
    # Nothing you just can't be warlorded
    return [option(name="bishop", perpetrator=agent.id)]

def cardinal_options(agent, game):
    options = []
    for player in game.players:
        # Check each card in our hand
        for card in agent.hand.cards:
            # If the card cost is less than the player's gold
            cost = card.cost
            factory = False
            replica = 0
            if Card(**{"suit":"unique", "type_ID":35, "cost": 6}) in agent.buildings.cards and card.suit == "unique":
                cost -= -1
                factory = True
            if card in agent.buildings.cards and Card(**{"suit":"unique", "type_ID":36, "cost": 5}) and not agent.replicas: 
                replica = agent.replicas+1
            if cost <= player.gold:
                # Calculate how many cards we need to give in exchange
                exchange_cards_count = max(player.gold - cost, 0)
                # If we have enough cards to give (excluding the current card)
                if len(agent.hand.cards) - 1 >= exchange_cards_count:
                    # Get all combinations of exchange_cards_count cards (excluding the current card)
                    other_cards = [c for c in agent.hand.cards if c != card]
                    exchange_combinations = list(combinations(other_cards, exchange_cards_count))
                    # Each combination of exchange cards is a possible trade
                    for i in range(0, len(exchange_combinations), max(round(len(exchange_combinations)/1e2), 1)):
                        options.append(option(name="cardinal_exchange", perpetrator=agent.id, target=player.id, built_card=card, cards_to_give=exchange_combinations[i], replica=replica, factory=factory))

    return options


def abbot_options(agent, game):
    options = []
    total_religious_districts = sum([1 if card.suit == "religion" else 0 for card in agent.hand.cards])
    if total_religious_districts > 0:
        gold_or_card_combinations = combinations_with_replacement(["gold", "card"], total_religious_districts)
        for gold_or_card_combination in gold_or_card_combinations:
            options.append(option(name="abbot_gold_or_card", perpetrator=agent.id,  gold_or_card_combination=list(gold_or_card_combination)))
        return options
    return []
    
def abbot_beg(agent, game):
    if not "begged" in game.gamestate.already_done_moves:
        return [option(name="abbot_beg", perpetrator=agent.id)]
    return []
    
#ID 5
def merchant_options(agent, game):
    # Nothing to choose
    return [option(name="merchant", perpetrator=agent.id)]
    
def alchemist_options(agent, game):
    # Nothing to choose
    return []
    
def trader_options(agent, game):
    # Nothing to choose
    return [option(name="trader", perpetrator=agent.id)]
    
# ID 6
def architect_options(agent, game):
    return [option(name="architect", perpetrator=agent.id)]

def navigator_options(agent, game):
    return [option(name="navigator_gold_card", perpetrator=agent.id, choice="4gold"), option(name="navigator_gold_card", perpetrator=agent.id, choice="4card")]
    
def scholar_options(agent, game):
    if game.deck.cards:
        return [option(name="scholar", perpetrator=agent.id)]
    return []
    
def scholar_give_back_options(agent, game):
    options = []
    seven_drawn_cards = game.seven_drawn_cards
    for card in seven_drawn_cards.cards:
        unchosen_cards = copy(seven_drawn_cards)
        unchosen_cards.get_a_card_like_it(card)
        options.append(option(name="scholar_card_pick", choice=card, perpetrator=agent.id,  unchosen_cards=unchosen_cards, chosen_card=card))

    return options
    
# ID 7
def warlord_options(agent, game):
    options = []
    for player in game.players:
        if len(player.buildings.cards) < 7:
            for building in player.buildings.cards:
                if building.cost-1 <= agent.gold and building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop":
                    if option(name="warlord_desctruction", target=player.id, perpetrator=agent.id, choice=building) not in options:
                        options.append(option(name="warlord_desctruction", target=player.id, perpetrator=agent.id, choice=building))
    
    return options
    
def marshal_options(agent, game):
    options = []
    for player in game.players:
        if len(player.buildings.cards) < 7 and player.id != agent.id:
            for building in player.buildings.cards:
                if building.cost <= agent.gold and building.cost <= 3 and building not in agent.buildings.cards and building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop":
                    if option(name="marshal_steal", target=player.id, perpetrator=agent.id, choice=building) not in options:
                        options.append(option(name="marshal_steal", target=player.id, perpetrator=agent.id, choice=building))
    return options
    
def diplomat_options(agent, game):
    options = []
    for player in game.players:
        if len(player.buildings.cards) < 7 and player.id != agent.id:
            for enemy_building in player.buildings.cards:
                for own_building in agent.buildings.cards:
                    if enemy_building.cost-own_building.cost <= agent.gold and enemy_building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop" and enemy_building not in agent.buildings.cards:
                        if option(name="diplomat_exchange", target=player.id, perpetrator=agent.id, choice=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)) not in options:
                            options.append(option(name="diplomat_exchange", target=player.id, perpetrator=agent.id, choice=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)))
    
    return options
    
def take_gold_for_war_options(agent, game):
    if "take_gold" not in game.gamestate.already_done_moves:
        return [option(name="take_gold_for_war", perpetrator=agent.id)]
    return []
# ID 8 later for more players


