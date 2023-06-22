from game.deck import Deck, Card
from game.option import option
from itertools import combinations, permutations, combinations_with_replacement
from copy import copy

class Agent():

    def __init__(self, id:int) -> None:
        # Game start
        self.hand = Deck(empty=True)
        self.role = None
        self.buildings = Deck(empty=True)
        self.warrant_fake = False
        self.warrant_true = False
        self.blackmail_fake = False
        self.blackmail_true = False
        self.crown = False
        self.money = 2
        self.id = id

        self.already_used_smithy = False
        self.already_used_lab = False
        self.already_used_museum = False

    def choose_role_options(self, avaible_roles:list) -> list:
        options = [option(choice=role, name="Role") for role in avaible_roles]
        return options
    
    def gold_or_card(self) -> list:
        options = [option(choice="gold", name="gold_or_card"), option(choice="gold", name="gold_or_card")]
        return options
    
    def blackmail_response_options(self, ) -> list:
        if self.blackmail_fake or self.blackmail_true:
            return [option(choice="pay", name="blackmail_response"), option(choice="not_pay", name="blackmail_response")]
        return [option(name="empty_option")]
    
    def ghost_town_color_choice_options(self) -> list:
        # Async
        if Card(**{"suit":"unique", "type_ID":19, "cost": 2}) in self.buildings.cards:
            return [option(choice="trade", name="ghost_town_color_choice"), option(choice="war", name="ghost_town_color_choice"),
                     option(choice="religion", name="ghost_town_color_choice"), option(choice="lord", name="ghost_town_color_choice"), option(choice="unique", name="ghost_town_color_choice")]
        return [option(name="empty_option")]
        
    def smithy_options(self) -> list:
        if Card(**{"suit":"unique", "type_ID":21, "cost": 5}) in self.buildings.cards and self.money >= 2 and not self.already_used_smithy:
            return [option(name="empty_option"), option(name="smithy_choice")]
        return []
    
    def laboratory_options(self) -> list:
        if Card(**{"suit":"unique", "type_ID":22, "cost": 5}) in self.buildings.cards and len(self.hand.cards) >= 1 and not self.already_used_lab:
            return [option(name="empty_option"), option(name="laboratory_choice")]
        return []
    
    def magic_school_options(self) -> list:
        # Every round
        # used before character ability
        if Card(**{"suit":"unique", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"trade", "type_ID":23, "cost": 6}) in self.buildings.cards or Card(**{"suit":"war", "type_ID":23, "cost": 6}) in self.buildings.cards or Card(**{"suit":"religion", "type_ID":23, "cost": 6}) in self.buildings.cards or Card(**{"suit":"lord", "type_ID":23, "cost": 6}) in self.buildings.cards:
            return [option(choice="trade", name="magic_school_choice"), option(choice="war", name="magic_school_choice"),
                     option(choice="religion", name="magic_school_choice"), option(choice="lord", name="magic_school_choice"), option(choice="unique", name="magic_school_choice")]
        return [option(name="empty_option")]
    
    def weapon_storage_options(self, players) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":27, "cost": 3}) in self.buildings.cards:
            for player in players:
                if player.id != self.id:
                    for card in player.buildings.cards:
                        options.append(option(who=player.id, choice = card.type_ID, name="weapon_storage_choice"))
        return options
    
    def lighthouse_options(self, game) -> list:
        # Async, HAVE TO USE RIGHT AFTER BUILDING THE HOUSE
        # SHUFFLE AFTER, AND KNOW WHATS LEFT IN THE DECK
        options = []
        if Card(**{"suit":"unique", "type_ID":29, "cost": 3}) in self.buildings.cards:
            for card in game.deck.cards:
                options.append(option(choice=card.type_ID, name="lighthouse_choice"))
        return options + [option(name="empty_option")]
    
    def museum_options(self) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":34, "cost": 4}) in self.buildings.cards and not self.already_used_museum:
            for card in self.hand.cards:
                options.append(option(choice=card.type_ID, name="museum_choice"))
        return options + [option(name="empty_option")]
    
    
    def character_options(self, game):
        pass
    
    # ID 0
    def assasin_options(self, game):
        options = []
        if self.role == "Assassin":
            for role_ID in game.roles:
                if role_ID > 0 and role_ID != next(iter(game.visible_face_up_role.keys())):
                    options.append(option(name="assassination", target=role_ID))
        return options
        
    def magistrate_options(self, game):
        options = []
        if self.role == "Magistrate":
            # [1:] to remove the magistrate itself
            target_possibilities = list(game.roles.keys())[1:]
            # Remove the visible left out role
            target_possibilities.remove(next(iter(game.visible_face_up_role.keys())))

            for real_target in target_possibilities:
                for fake_tagets in combinations(target_possibilities, 2):
                    if real_target not in fake_tagets:
                        options.append(option(name="magistrate_warrant", real_target=real_target, fake_targets=list(fake_tagets)))
        return options

    def witch_options(self, game):
        options = []
        if self.role == "Witch":
            for role_ID in game.roles:
                if role_ID > 0 and role_ID != next(iter(game.visible_face_up_role.keys())):
                    options.append(option(name="bewitching", target=role_ID))
        return options
    
    # ID 1
    def thief_options(self, game):
        options = []
        if self.role == "Thief":
            for role_ID in game.roles:
                if role_ID > 1 and role_ID != next(iter(game.visible_face_up_role.keys())):
                    options.append(option(name="steal", target=role_ID))
        return options
    
    def blackmail_options(self, game):
        options = []
        if self.role == "Blackmailer":
            # [2:] to remove the blackmailer and the ID 0 character (assassin and the like)
            target_possibilities = list(game.roles.keys())[2:]
            # Remove the visible face up role
            target_possibilities.remove(next(iter(game.visible_face_up_role.keys())))
            
            for targets in combinations(target_possibilities, 2):
                options.append(option(name="blackmail", real_target=targets[0], fake_target=targets[1]))
                options.append(option(name="blackmail", real_target=targets[1], fake_target=targets[0]))
        return options
    
    def spy_options(self, game):
        options = []
        if self.role == "Spy":
            for role_ID in game.roles:
                for suit in ["trade", "war", "religion", "lord", "unique"]:
                    if role_ID > 1 and role_ID != next(iter(game.visible_face_up_role.keys())):
                        options.append(option(name="spy", target=role_ID, suit=suit))
        return options
    
    # ID 2
    def magician_options(self, game):
        options = []
        if self.role == "Magician":
            for role_ID in game.roles:
                if role_ID != 2 and role_ID != next(iter(game.visible_face_up_role.keys())):
                    options.append(option(name="magic_hand_change", target=role_ID))
                    
            for r in range(1, len(self.hand.cards)+1):
                # list of cards
                discard_possibilities = list(combinations(self.hand.cards, r))
                for discard_possibility in discard_possibilities:
                    options.append(option(name="discard_and_draw", cards=discard_possibility))
            
        return options
    
    def wizard_look_at_hand_options(self, game):
        options = []
        if self.role == "Wizard":
            for player in game.players:
                if player.id != self.id:
                    options.append(option(name="look_at_hand", target_player= player.id))
        return options
    
    def wizard_take_from_hand_options(self, target_player):
        # TODO can build immediatly and can be the same building
        options = []
        if self.role == "Wizard":
            for card in target_player.hand.cards:
                options.append(option(name="take_from_hand", card=card))
        return options
    
    def seer_give_back_card(self, players_with_taken_cards):
        options = []
        if self.role == "Seer":
            for permutation in permutations(self.hand.cards, len(players_with_taken_cards)):
                card_handouts = {player_card_pair[0] : player_card_pair[1] for player_card_pair in zip(players_with_taken_cards, permutation)}
                options.append(option(name="give_back_card", card_handouts=card_handouts))
        return options

    # ID 3
    def king_options(self, game):
        # Nothing you just take the crown
        return []

    def emperor_options(self, game):
        options = []
        if self.role == "Emperor":
            for player in game.players:
                if player.id != self.id:
                    if len(player.hand.cards):
                        options.append(option(name="give_crown", target=player, gold_or_card="card"))
                    if player.gold:
                        options.append(option(name="give_crown", target=player, gold_or_card="gold"))
                    if not player.gold and not len(player.hand.cards):
                        options.append(option(name="give_crown", target=player, gold_or_card="nothing"))
                    
        return options
    
    def patrician_options(self, game):
        # Nothing you just take the crown
        return []

    # ID 4
    def bishop_options(self, game):
        # Nothing you just can't be warlorded
        return []

    def cardinal_options(self, game):
        options = []
        if self.role == "Cardinal":

            for player in game.players:
                # Check each card in our hand
                for card in self.hand:
                    # If the card cost is less than the player's gold
                    if card.cost <= player.gold:
                        # Calculate how many cards we need to give in exchange
                        exchange_cards_count = player.gold - card.cost

                        # If we have enough cards to give (excluding the current card)
                        if len(self.hand) - 1 >= exchange_cards_count:
                            # Get all combinations of exchange_cards_count cards (excluding the current card)
                            other_cards = [c for c in self.hand if c != card]
                            exchange_combinations = combinations(other_cards, exchange_cards_count)

                            # Each combination of exchange cards is a possible trade
                            for exchange_cards in exchange_combinations:
                                options.append(option(name="cardinal_exchange", target_player=player, built_card=card, cards_to_give=exchange_cards))

        return options
    
    def abbot_options(self, game):
        options = []
        if self.role == "Abbot":
            total_religious_districts = sum([1 if card.suit == "religion" else 0 for card in self.hand])
            gold_or_card_combinations = combinations_with_replacement(["gold", "card"], total_religious_districts)
            for gold_or_card_combination in gold_or_card_combinations:
                options.append(option(name="abbot_gold_or_card", gold_or_card_combination=list(gold_or_card_combination)))
            
        return options
    
    #ID 5
    def merchant_options(self, game):
        # Nothing to choose
        return []
    
    def alchemist_options(self, game):
        # Nothing to choose
        return []
    
    def trader_options(self, game):
        # Nothing to choose
        return []
    
    # ID 6
    def architect_options(self, game):
        # Nothing to choose
        return []
    
    def navigator_options(self, game):
        return [option(name="navigator_gold_card", choice="4gold"), option(name="navigator_gold_card", choice="4card")]

    def scholar_options(self, seven_drawn_cards):
        options = []
        if self.role == "Scholar":
            for card in seven_drawn_cards:
                unchosen_cards = copy(seven_drawn_cards)
                unchosen_cards.remove(card)
                options.append(option(name="scholar_card_pick", choice=card, unchosen_cards=unchosen_cards))
            return options
        
    # ID 7
    def warlord_options(self, game):
        options = []
        if self.role == "Warlord":
            for player in game.players:
                if len(player.buildings.cards) < 7:
                    for building in player.buildings.cards:
                        if building.cost-1 <= self.gold:
                            if option(name="warlord_desctruction", target=player, choice=building) not in options:
                                options.append(option(name="warlord_desctruction", target=player, choice=building))
                            
        return options
    
    def marshal_options(self, game):
        options = []
        if self.role == "Marshal":
            for player in game.players:
                if len(player.buildings.cards) < 7 and player.id != self.id:
                    for building in player.buildings.cards:
                        if building.cost <= self.gold and building.cost <= 3:
                            if option(name="marshal_steal", target=player, choice=building) not in options:
                                options.append(option(name="marshal_steal", target=player, choice=building))
                                
    def diplomat_options(self, game):
        options = []
        if self.role == "Diplomat":
            for player in game.players:
                if len(player.buildings.cards) < 7 and player.id != self.id:
                    for enemy_building in player.buildings.cards:
                        for own_building in self.buildings.cards:
                            if enemy_building.cost-own_building.cost <= self.gold:
                                if option(name="diplomat_exchange", target=player, take=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)) not in options:
                                    options.append(option(name="diplomat_exchange", target=player, take=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)))
                                    
    # ID 8 later for more players
        

