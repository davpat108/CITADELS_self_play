from game.deck import Deck, Card
from game.option import option
from game.config import role_to_role_id
from itertools import combinations, permutations, combinations_with_replacement
from copy import copy
from game.helper_classes import HandKnowledge, GameState, RoleKnowlage

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
        return [option(name="role_pick", perpetrator=self.id, choice=role) for role in game.roles_to_choose_from.values()]

    def gold_or_card_options(self, game):
        return [option(name="gold_or_card", perpetrator=self.id, choice="gold"), option(name="gold_or_card", perpetrator=self.id, choice="card")] if len(game.deck.cards) > 1 else [option(name="gold_or_card", perpetrator=self.id, choice="gold")]
    
    def which_card_to_keep_options(self, game):
        options = []
        if Card(**{"suit":"unique", "type_ID":20, "cost": 4}) in self.buildings.cards:
            card_choices = list(combinations(self.just_drawn_cards.cards, 2))
            for card_choice in card_choices:
                options.append(option(name="which_card_to_keep", perpetrator=self.id, choice=card_choice))
            return options
        return [option(name="which_card_to_keep", perpetrator=self.id, choice=[card]) for card in self.just_drawn_cards.cards]

    def blackmail_response_options(self, game) -> list:
        if game.role_properties[role_to_role_id[self.role]].blackmail:
            return [option(choice="pay", perpetrator=self.id, name="blackmail_response"), option(choice="not_pay", perpetrator=self.id, name="blackmail_response")]
        return [option(name="empty_option", perpetrator=self.id, next_gamestate=GameState(state=5, player_id=self.id))]
    
    def reveal_blackmail_as_blackmailer_options(self, game) -> list:
        return [option(choice="reveal", perpetrator=self.id, target=game.gamestate.next_gamestate.player_id, name="reveal_blackmail_as_blackmailer"), option(choice="not_reveal", perpetrator=self.id, target=game.gamestate.next_gamestate.player_id, name="reveal_blackmail_as_blackmailer")]
    
    def reveal_warrant_as_magistrate_options(self, game) -> list:
        return [option(choice="reveal", perpetrator=self.id, target=game.gamestate.next_gamestate.player_id, name="reveal_warrant_as_magistrate"), option(choice="not_reveal", perpetrator=self.id, target=game.gamestate.next_gamestate.player_id, name="reveal_warrant_as_magistrate")]


    def ghost_town_color_choice_options(self) -> list:
        # Async
        if Card(**{"suit":"unique", "type_ID":19, "cost": 2}) in self.buildings.cards:
            return [option(choice="trade", perpetrator=self.id, name="ghost_town_color_choice"), option(choice="war", perpetrator=self.id, name="ghost_town_color_choice"),
                     option(choice="religion", perpetrator=self.id, name="ghost_town_color_choice"), option(choice="lord", perpetrator=self.id, name="ghost_town_color_choice"), option(choice="unique", perpetrator=self.id, name="ghost_town_color_choice")]
        return []
        
    def smithy_options(self, game) -> list:
        if Card(**{"suit":"unique", "type_ID":21, "cost": 5}) in self.buildings.cards and self.gold >= 2 and not "smithy" in game.gamestate.already_done_moves:
            return [option(name="smithy_choice", perpetrator=self.id,)]
        return []
    
    def laboratory_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":22, "cost": 5}) in self.buildings.cards and not "lab" in game.gamestate.already_done_moves:
            for card in self.hand.cards:
                options.append(option(name="laboratory_choice", perpetrator=self.id, choice=card))
        return []
    
    def magic_school_options(self, game) -> list:
        # Every round
        # used before character ability
        if not "magic_school" in game.gamestate.already_done_moves:
            if Card(**{"suit":"unique", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"trade", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"war", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"religion", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"lord", "type_ID":25, "cost": 6}) in self.buildings.cards:
                return [option(choice="trade", perpetrator=self.id, name="magic_school_choice"), option(choice="war", perpetrator=self.id, name="magic_school_choice"),
                         option(choice="religion", perpetrator=self.id, name="magic_school_choice"), option(choice="lord", perpetrator=self.id, name="magic_school_choice"), option(choice="unique", perpetrator=self.id, name="magic_school_choice")]
        return []
    
    def weapon_storage_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":27, "cost": 3}) in self.buildings.cards:
            for player in game.players:
                if player.id != self.id:
                    for card in player.buildings.cards:
                        options.append(option(perpetrator=self.id,target=player.id, choice = card, name="weapon_storage_choice"))
        return options
    
    def lighthouse_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":29, "cost": 3}) in self.buildings.cards and self.can_use_lighthouse:
            for card in game.deck.cards:
                options.append(option(choice=card, perpetrator=self.id, name="lighthouse_choice"))
        return options

    def museum_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":34, "cost": 4}) in self.buildings.cards and not "museum" in game.gamestate.already_done_moves:
            for card in self.hand.cards:
                options.append(option(choice=card, perpetrator=self.id, name="museum_choice"))
        return options

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
        return options
        
    
    def main_round_options(self, game):
        options = [option(name="finish_round", perpetrator=self.id, next_witch=False, crown=False)]
        options += self.build_options(game)
        options += self.character_options(game)
        options += self.smithy_options(game)
        options += self.laboratory_options(game)
        options += self.magic_school_options(game)
        options += self.weapon_storage_options(game)
        options += self.lighthouse_options(game)
        options += self.museum_options(game)
        return options
    
    def graveyard_options(self, game):
        if self.gold > 0:
            return [option(name="graveyard", perpetrator=self.id)]
        return [option(name="empty_option", perpetrator=self.id, next_gamestate=game.gamestate.next_gamestate)]

    def character_options(self, game):
        options = []
        # ID 0
        if "character_ability" not in game.gamestate.already_done_moves:
            if self.role == "Assassin":
                options += self.assasin_options(game)
            elif self.role == "Magistrate":
                options += self.magistrate_options(game)

            #ID 1
            elif self.role == "Thief":
                options += self.thief_options(game)
            elif self.role == "Blackmailer":
                options += self.blackmail_options(game)
            elif self.role == "Spy":
                options += self.spy_options(game)

            #ID 2
            elif self.role == "Magician":
                options += self.magician_options(game)
            elif self.role == "Wizard":
                options += self.wizard_look_at_hand_options(game)
            elif self.role == "Seer":
                options += self.seer_options(game)

            #ID 3
            elif self.role == "King":
                options += self.king_options(game)
            elif self.role == "Emperor":
                options += self.emperor_options(game)
            elif self.role == "Patrician":
                options += self.patrician_options(game)

            #ID 4
            elif self.role == "Bishop":
                options += self.bishop_options(game)
            elif self.role == "Cardinal":
                options += self.cardinal_options(game)
            elif self.role == "Abbot":
                options += self.abbot_options(game)

            #ID 5
            elif self.role == "Merchant":
                options += self.merchant_options(game)
            elif self.role == "Alchemist":
                options += self.alchemist_options(game)
            elif self.role == "Trader":
                options += self.trader_options(game)

            #ID 6
            elif self.role == "Architect":
                options += self.architect_options(game)
            elif self.role == "Navigator":
                options += self.navigator_options(game)
            elif self.role == "Scholar":
                options += self.scholar_options(game)

            #ID 7
            elif self.role == "Warlord":
                options += self.warlord_options(game)
            elif self.role == "Marshal":
                options += self.marshal_options(game)
            elif self.role == "Diplomat":
                options += self.diplomat_options(game)
        
        elif self.role == "Warlord" or self.role == "Marshal" or self.role == "Diplomat":
            options += self.take_gold_for_war_options(game)
        elif self.role == "Abbot":
            options += self.abbot_beg(game)
        #ID 8
        # Not yet

        return options


    # ID 0
    def assasin_options(self, game):
        target_possibilities = game.roles.copy()
        if game.visible_face_up_role:
            target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
        return [option(name="assassination", perpetrator=self.id, target=role_ID) for role_ID in target_possibilities.keys() if role_ID > 0]
        
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
        return options

    def witch_options(self, game):
        target_possibilities = game.roles.copy()
        if game.visible_face_up_role:
            target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
        return [option(name="bewitching",  perpetrator=self.id, target=role_ID) for role_ID in target_possibilities.keys() if role_ID > 0]
    
    # ID 1
    def thief_options(self, game):
        target_possibilities = game.roles.copy()
        if game.visible_face_up_role:
            target_possibilities.pop(next(iter(game.visible_face_up_role.keys())))
        return [option(name="steal", perpetrator=self.id, target=role_ID) for role_ID in target_possibilities.keys() if role_ID > 1]
    
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
        return options
    
    def spy_options(self, game):
        options = []
        for player in game.players:
            if player.id != self.id:
                for suit in ["trade", "war", "religion", "lord", "unique"]:
                    options.append(option(name="spy", perpetrator=self.id, target=player.id, suit=suit))
        return options
    
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
            
        return options
    
    def wizard_look_at_hand_options(self, game):
        options = []
        for player in game.players:
            if player.id != self.id and len(player.hand.cards) > 0:
                options.append(option(name="look_at_hand", perpetrator=self.id, target=player.id))
        return options
    
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
        return options
    
    def seer_options(self, game):
        return [option(name="seer", perpetrator=self.id)]
    

    def seer_give_back_card(self, game):
        options = []
        perms = list(permutations(self.hand.cards, len(game.seer_taken_card_from)))
        for i in range(0, len(perms), max(round(len(perms)/1e3), 1)):
            card_handouts = {player_card_pair[0] : player_card_pair[1] for player_card_pair in zip(game.seer_taken_card_from, perms[i])}
            options.append(option(name="give_back_card", perpetrator=self.id, card_handouts=card_handouts))
        return options

    # ID 3
    def king_options(self, game):
        # Nothing you just take the crown
        return [option(name="take_crown_king", perpetrator=self.id)]

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
                    
        return options
    
    def patrician_options(self, game):
        # Nothing you just take the crown
        return [option(name="take_crown_pat", perpetrator=self.id)]

    # ID 4
    def bishop_options(self, game):
        # Nothing you just can't be warlorded
        return [option(name="bishop", perpetrator=self.id)]

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

        return options
    
    def abbot_options(self, game):
        options = []
        total_religious_districts = sum([1 if card.suit == "religion" else 0 for card in self.hand.cards])
        gold_or_card_combinations = combinations_with_replacement(["gold", "card"], total_religious_districts)
        for gold_or_card_combination in gold_or_card_combinations:
            options.append(option(name="abbot_gold_or_card", perpetrator=self.id,  gold_or_card_combination=list(gold_or_card_combination)))
        return options
    
    def abbot_beg(self, game):
        if not "begged" in game.gamestate.already_done_moves:
            return [option(name="abbot_beg", perpetrator=self.id)]
        return []
    
    #ID 5
    def merchant_options(self, game):
        # Nothing to choose
        return [option(name="merchant", perpetrator=self.id)]
    
    def alchemist_options(self, game):
        # Nothing to choose
        return []
    
    def trader_options(self, game):
        # Nothing to choose
        return [option(name="trader", perpetrator=self.id)]
    
    # ID 6
    def architect_options(self, game):
        return [option(name="architect", perpetrator=self.id)]
    
    def navigator_options(self, game):
        return [option(name="navigator_gold_card", perpetrator=self.id, choice="4gold"), option(name="navigator_gold_card", perpetrator=self.id, choice="4card")]

    def scholar_options(self, game):
        if game.deck.cards:
            return [option(name="scholar", perpetrator=self.id)]
        return []

    def scholar_give_back_options(self, game):
        options = []
        seven_drawn_cards = game.seven_drawn_cards
        for card in seven_drawn_cards.cards:
            unchosen_cards = copy(seven_drawn_cards)
            unchosen_cards.get_a_card_like_it(card)
            options.append(option(name="scholar_card_pick", choice=card, perpetrator=self.id,  unchosen_cards=unchosen_cards))
        return options

    # ID 7
    def warlord_options(self, game):
        options = []
        for player in game.players:
            if len(player.buildings.cards) < 7:
                for building in player.buildings.cards:
                    if building.cost-1 <= self.gold and building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop":
                        if option(name="warlord_desctruction", target=player.id, perpetrator=self.id, choice=building) not in options:
                            options.append(option(name="warlord_desctruction", target=player.id, perpetrator=self.id, choice=building))
                            
        return options

    def marshal_options(self, game):
        options = []
        for player in game.players:
            if len(player.buildings.cards) < 7 and player.id != self.id:
                for building in player.buildings.cards:
                    if building.cost <= self.gold and building.cost <= 3 and building not in self.buildings.cards and building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop":
                        if option(name="marshal_steal", target=player.id, perpetrator=self.id, choice=building) not in options:
                            options.append(option(name="marshal_steal", target=player.id, perpetrator=self.id, choice=building))
        return options    

    def diplomat_options(self, game):
        options = []
        for player in game.players:
            if len(player.buildings.cards) < 7 and player.id != self.id:
                for enemy_building in player.buildings.cards:
                    for own_building in self.buildings.cards:
                        if enemy_building.cost-own_building.cost <= self.gold and enemy_building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}) and player.role != "Bishop" and enemy_building not in self.buildings.cards:
                            if option(name="diplomat_exchange", target=player.id, perpetrator=self.id, choice=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)) not in options:
                                options.append(option(name="diplomat_exchange", target=player.id, perpetrator=self.id, choice=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)))
        return options  
    
    def take_gold_for_war_options(self, game):
        if "take_gold" not in game.gamestate.already_done_moves:
            return [option(name="take_gold_for_war", perpetrator=self.id)]
        return []
    # ID 8 later for more players
    
