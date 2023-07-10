from game.deck import Deck, Card
from game.option import option
from game.config import role_to_role_id
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
        self.dead = False

        self.just_drawn_cards = Deck(empty=True)
        self.has_replica = False
        
        self.museum_cards = Deck(empty=True)

        self.crown = False
        self.gold = 2
        self.id = id


    def get_options(self, game):
        if game.game_state.state == 0:
            return self.pick_role_options(game)
        elif game.game_state.state == 1:
            return self.gold_or_card_options(game)
        elif game.game_state.state == 2:
            return self.which_card_to_keep_options(game)
        elif game.game_state.state == 3:
            return self.blackmail_response_options(game)
        elif game.game_state.state == 4:
            return self.reveal_blackmail_as_blackmailer_options(game)
        elif game.game_state.state == 5:
            return self.main_round_options(game)
        elif game.game_state.state == 6:
            return self.graveyard_options(game)

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
    
    def count_points(self, game):
        points = 0
        for card in self.buildings.cards:
            points += card.cost
            if card.type_ID == 18 or card.type_ID == 23:
                points += 2
            # Whishing well
            if Card(**{"suit":"unique", "type_ID":31, "cost": 5}) in self.buildings.cards and card.suit == "unique":
                points += 1
        
        points += len(self.museum_cards)

        # Imp teasury
        if Card(**{"suit":"unique", "type_ID":37, "cost": 4}) in self.buildings.cards:
            points += self.gold
        
        # Maproom
        if Card(**{"suit":"unique", "type_ID":39, "cost": 5}) in self.buildings.cards:
            points += len(self.hand.cards)


    # Options from gamestates
    def pick_role_options(self, game):
        return [option(name="role_pick", perpetrator=self, choice=role) for role in game.roles_to_choose_from.values()]

    def gold_or_card_options(self, game):
        return [option(name="gold_or_card", perpetrator=self, choice="gold"), option(name="gold_or_card", perpetrator=self, choice="card")]
    
    def which_card_to_keep_options(self, game):
        options = []
        if Card(**{"suit":"unique", "type_ID":20, "cost": 4}) in self.buildings.cards:
            card_choices = list(combinations(self.just_drawn_cards.cards, 2))
            for card_choice in card_choices:
                options.append(option(name="which_card_to_keep", perpetrator=self, choice=card_choice))
            return options
        return [option(name="which_card_to_keep", perpetrator=self, choice=[card]) for card in self.just_drawn_cards.cards]

    def blackmail_response_options(self, game) -> list:
        if game.role_properties[role_to_role_id[self.role]].blackmail:
            return [option(choice="pay", perpetrator=self, name="blackmail_response"), option(choice="not_pay", perpetrator=self, name="blackmail_response")]
        return [option(name="empty_option", perpetrator=self, next_gamestate=4, next_player=self)]
    
    def reveal_blackmail_as_blackmailer_options(self, game) -> list:
        return [option(choice="reveal", perpetrator=self, target=game.blackmailed_player, name="reveal_blackmail_as_blackmailer"), option(choice="not_reveal", perpetrator=self, target=game.blackmailed_player, name="reveal_blackmail_as_blackmailer")]
    

    def ghost_town_color_choice_options(self) -> list:
        # Async
        if Card(**{"suit":"unique", "type_ID":19, "cost": 2}) in self.buildings.cards:
            return [option(choice="trade", perpetrator=self, name="ghost_town_color_choice"), option(choice="war", perpetrator=self, name="ghost_town_color_choice"),
                     option(choice="religion", perpetrator=self, name="ghost_town_color_choice"), option(choice="lord", perpetrator=self, name="ghost_town_color_choice"), option(choice="unique", perpetrator=self, name="ghost_town_color_choice")]
        return []
        
    def smithy_options(self, game) -> list:
        if Card(**{"suit":"unique", "type_ID":21, "cost": 5}) in self.buildings.cards and self.gold >= 2 and not "smithy" in game.game_state.already_done_moves:
            return [option(name="smithy_choice", perpetrator=self,)]
        return []
    
    def laboratory_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":22, "cost": 5}) in self.buildings.cards and not "lab" in game.game_state.already_done_moves:
            for card in self.hand.cards:
                options.append(option(name="laboratory_choice", perpetrator=self, choice=card))
        return []
    
    def magic_school_options(self, game) -> list:
        # Every round
        # used before character ability
        if not "magic_school" in game.game_state.already_done_moves:
            if Card(**{"suit":"unique", "type_ID":25, "cost": 6}) in self.buildings.cards or Card(**{"suit":"trade", "type_ID":23, "cost": 6}) in self.buildings.cards or Card(**{"suit":"war", "type_ID":23, "cost": 6}) in self.buildings.cards or Card(**{"suit":"religion", "type_ID":23, "cost": 6}) in self.buildings.cards or Card(**{"suit":"lord", "type_ID":23, "cost": 6}) in self.buildings.cards:
                return [option(choice="trade", perpetrator=self, name="magic_school_choice"), option(choice="war", perpetrator=self, name="magic_school_choice"),
                         option(choice="religion", perpetrator=self, name="magic_school_choice"), option(choice="lord", perpetrator=self, name="magic_school_choice"), option(choice="unique", perpetrator=self, name="magic_school_choice")]
        return []
    
    def weapon_storage_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":27, "cost": 3}) in self.buildings.cards:
            for player in game.players:
                if player.id != self.id:
                    for card in player.buildings.cards:
                        options.append(option(perpetrator=self,target=player, choice = card, name="weapon_storage_choice"))
        return options
    
    def lighthouse_options(self, game) -> list:
        # Async, HAVE TO USE RIGHT AFTER BUILDING THE HOUSE
        # SHUFFLE AFTER, AND KNOW WHATS LEFT IN THE DECK
        options = []
        if Card(**{"suit":"unique", "type_ID":29, "cost": 3}) in self.buildings.cards:
            for card in game.deck.cards:
                options.append(option(choice=card.type_ID, perpetrator=self, name="lighthouse_choice"))
        return options

    def museum_options(self, game) -> list:
        options = []
        if Card(**{"suit":"unique", "type_ID":34, "cost": 4}) in self.buildings.cards and not "museum" in game.game_state.already_done_moves:
            for card in self.hand.cards:
                options.append(option(choice=card, perpetrator=self, name="museum_choice"))
        return options

    # Main stuff
    def get_builds(self, options) -> list:
        # Returns buildable cards from hand by cost
        for card in self.hand:
            cost = card.cost
            factory = False
            if Card(**{"suit":"unique", "type_ID":35, "cost": 6}) in self.buildings and card.suit == "unique":
                cost -= -1
                factory = True
            if card in self.buildings.cards and Card(**{"suit":"unique", "type_ID":36, "cost": 5}) and not self.has_replica: 
                replica = True
            if cost <= self.gold and option(name="build", perpetrator=self, built_card=card, replica=replica, factory=factory) not in options:
                options.append(option(name="build", perpetrator=self, built_card=card, replica=replica, factory=factory))


    def build_options(self, game):
        options = []
        build_limit = self.get_build_limit()
        if self.role != "Trader":
            if game.game_state.already_done_moves.count("trade_building") + game.game_state.already_done_moves.count("non_trade_building") < build_limit:
                self.get_builds(options)
        else:
            if game.game_state.already_done_moves.count("non_trade_building") < build_limit:
                self.get_builds(options)
        return options
        
    
    def main_round_options(self, game):
        options = []
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
            return [option(name="graveyard", perpetrator=self)]
        return []

    def character_options(self, game):
        options = []
        # ID 0
        if self.role == "Assassin":
            options += self.assasin_options(game)
        elif self.role == "Magistrate":
            options += self.magistrate_options(game)
        elif self.role == "Witch":
            options += self.witch_options(game)

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

        #ID 8
        # Not yet

        return options


    # ID 0
    def assasin_options(self, game):
        options = []
        if self.role == "Assassin":
            for role_ID in game.roles:
                if not (game.visible_face_up_role and role_ID > 0) and role_ID != next(iter(game.visible_face_up_role.keys())):
                    options.append(option(name="assassination", target=role_ID))
        return options
        
    def magistrate_options(self, game):
        options = []
        if self.role == "Magistrate":
            # [1:] to remove the magistrate itself
            target_possibilities = list(game.roles.keys())[1:]
            # Remove the visible left out role
            if game.visible_face_up_role:
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
                if not (game.visible_face_up_role and role_ID > 0) and role_ID != next(iter(game.visible_face_up_role.keys())):
                    options.append(option(name="bewitching", perpetrator=self, target=role_ID))
        return options
    
    # ID 1
    def thief_options(self, game):
        options = []
        if self.role == "Thief":
            for role_ID in game.roles:
                if not (game.visible_face_up_role and role_ID > 1) and role_ID != next(iter(game.visible_face_up_role.keys())):
                    options.append(option(name="steal", perpetrator=self, target=role_ID))
        return options
    
    def blackmail_options(self, game):
        options = []
        if self.role == "Blackmailer":
            # [2:] to remove the blackmailer and the ID 0 character (assassin and the like)
            target_possibilities = list(game.roles.keys())[2:]
            # Remove the visible face up role
            if game.visible_face_up_role:
                target_possibilities.remove(next(iter(game.visible_face_up_role.keys())))
            
            for targets in combinations(target_possibilities, 2):
                options.append(option(name="blackmail", perpetrator=self, real_target=targets[0], fake_target=targets[1]))
                options.append(option(name="blackmail", perpetrator=self, real_target=targets[1], fake_target=targets[0]))
        return options
    
    def spy_options(self, game):
        options = []
        if self.role == "Spy":
            for player in game.players:
                if player.id != self.id:
                    for suit in ["trade", "war", "religion", "lord", "unique"]:
                        options.append(option(name="spy", perpetrator=self, target=player, suit=suit))
        return options
    
    # ID 2
    def magician_options(self, game):
        options = []
        if self.role == "Magician":
            for player in game.players:
                if player.id != self.id:
                    options.append(option(name="magic_hand_change", perpetrator=self, target=player))
                    
            for r in range(1, len(self.hand.cards)+1):
                # list of cards
                discard_possibilities = list(combinations(self.hand.cards, r))
                for discard_possibility in discard_possibilities:
                    options.append(option(name="discard_and_draw", perpetrator=self, cards=discard_possibility))
            
        return options
    
    def wizard_look_at_hand_options(self, game):
        options = []
        if self.role == "Wizard":
            for player in game.players:
                if player.id != self.id:
                    options.append(option(name="look_at_hand", perpetrator=self, target_player=player))
        return options
    
    def wizard_take_from_hand_options(self, target_player):
        # TODO can build immediatly and can be the same building
        options = []
        if self.role == "Wizard":
            for card in target_player.hand.cards:
                options.append(option(name="take_from_hand", card=card, build=False, perpetrator=self, target_player=target_player))
                if self.gold >= card.cost:
                    options.append(option(name="take_from_hand", card=card, build=True, perpetrator=self, target_player=target_player))
        return options
    
    def seer_options(self, game):
        return [option(name="seer", perpetrator=self)]
    

    def seer_give_back_card(self, players_with_taken_cards):
        options = []
        if self.role == "Seer":
            for permutation in permutations(self.hand.cards, len(players_with_taken_cards)):
                card_handouts = {player_card_pair[0] : player_card_pair[1] for player_card_pair in zip(players_with_taken_cards, permutation)}
                options.append(option(name="give_back_card", perpetrator=self, card_handouts=card_handouts))
        return options

    # ID 3
    def king_options(self, game):
        # Nothing you just take the crown
        return [option(name="take_crown_king", perpetrator=self)]

    def emperor_options(self, game):
        options = []
        if self.role == "Emperor":
            for player in game.players:
                if player.id != self.id:
                    if len(player.hand.cards):
                        options.append(option(name="give_crown", perpetrator=self, target=player, gold_or_card="card"))
                    if player.gold:
                        options.append(option(name="give_crown", perpetrator=self, target=player, gold_or_card="gold"))
                    if not player.gold and not len(player.hand.cards):
                        options.append(option(name="give_crown", perpetrator=self,  target=player, gold_or_card="nothing"))
                    
        return options
    
    def patrician_options(self, game):
        # Nothing you just take the crown
        return [option(name="take_crown_pat", perpetrator=self)]

    # ID 4
    def bishop_options(self, game):
        # Nothing you just can't be warlorded
        return [option(name="bishop", perpetrator=self)]

    def cardinal_options(self, game):
        options = []
        if self.role == "Cardinal":
            for player in game.players:
                # Check each card in our hand
                for card in self.hand:
                    # If the card cost is less than the player's gold
                    cost = card.cost
                    factory = False
                    if Card(**{"suit":"unique", "type_ID":35, "cost": 6}) in self.buildings and card.suit == "unique":
                        cost -= -1
                        factory = True
                    if card in self.buildings.cards and Card(**{"suit":"unique", "type_ID":36, "cost": 5}) and not self.has_replica: 
                        replica = True
                    if cost <= player.gold:
                        # Calculate how many cards we need to give in exchange
                        exchange_cards_count = max(player.gold - cost, 0)

                        # If we have enough cards to give (excluding the current card)
                        if len(self.hand) - 1 >= exchange_cards_count:
                            # Get all combinations of exchange_cards_count cards (excluding the current card)
                            other_cards = [c for c in self.hand if c != card]
                            exchange_combinations = combinations(other_cards, exchange_cards_count)

                            # Each combination of exchange cards is a possible trade
                            for exchange_cards in exchange_combinations:
                                options.append(option(name="cardinal_exchange", perpetrator=self, target_player=player, built_card=card, cards_to_give=exchange_cards, replica=replica, factory=factory))

        return options
    
    def abbot_options(self, game):
        options = []
        if self.role == "Abbot":
            total_religious_districts = sum([1 if card.suit == "religion" else 0 for card in self.hand])
            gold_or_card_combinations = combinations_with_replacement(["gold", "card"], total_religious_districts)
            for gold_or_card_combination in gold_or_card_combinations:
                options.append(option(name="abbot_gold_or_card", gold_or_card_combination=list(gold_or_card_combination)))
            
        return options
    
    def abbot_beg(self, game):
        if not "begged" in game.game_state.already_done_moves:
            return [option(name="abbot_beg", perpetrator=self)]
        return []
    
    #ID 5
    def merchant_options(self, game):
        # Nothing to choose
        return [option(name="merchant", perpetrator=self)]
    
    def alchemist_options(self, game):
        # Nothing to choose
        return [option(name="alchemist", perpetrator=self)]
    
    def trader_options(self, game):
        # Nothing to choose
        return [option(name="trader", perpetrator=self)]
    
    # ID 6
    def architect_options(self, game):
        return [option(name="architect", perpetrator=self)]
    
    def navigator_options(self, game):
        return [option(name="navigator_gold_card", perpetrator=self, choice="4gold"), option(name="navigator_gold_card", perpetrator=self, choice="4card")]

    def scholar_options(self, game):
        return [option(name="scholar", perpetrator=self)]

    def scholar_give_back_options(self, seven_drawn_cards):
        options = []
        if self.role == "Scholar":
            for card in seven_drawn_cards:
                unchosen_cards = copy(seven_drawn_cards)
                unchosen_cards.remove(card)
                options.append(option(name="scholar_card_pick", choice=card, perpetrator=self,  unchosen_cards=unchosen_cards))
            return options
        
    # ID 7
    def warlord_options(self, game):
        options = []
        if self.role == "Warlord":
            for player in game.players:
                if len(player.buildings.cards) < 7:
                    for building in player.buildings.cards:
                        if building.cost-1 <= self.gold and building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}):
                            if option(name="warlord_desctruction", target=player, perpetrator=self, choice=building) not in options:
                                options.append(option(name="warlord_desctruction", target=player, perpetrator=self, choice=building))
                            
        return options
    
    def marshal_options(self, game):
        options = []
        if self.role == "Marshal":
            for player in game.players:
                if len(player.buildings.cards) < 7 and player.id != self.id:
                    for building in player.buildings.cards:
                        if building.cost <= self.gold and building.cost <= 3 and building not in self.buildings.cards and building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}):
                            if option(name="marshal_steal", target=player, perpetrator=self, choice=building) not in options:
                                options.append(option(name="marshal_steal", target=player, perpetrator=self, choice=building))
                                
    def diplomat_options(self, game):
        options = []
        if self.role == "Diplomat":
            for player in game.players:
                if len(player.buildings.cards) < 7 and player.id != self.id:
                    for enemy_building in player.buildings.cards:
                        for own_building in self.buildings.cards:
                            if enemy_building.cost-own_building.cost <= self.gold and enemy_building != Card(**{"suit":"unique", "type_ID":17, "cost": 3}):
                                if option(name="diplomat_exchange", target=player, perpetrator=self, take=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)) not in options:
                                    options.append(option(name="diplomat_exchange", target=player, perpetrator=self, take=enemy_building, give=own_building, money_owed=abs(enemy_building.cost-own_building.cost)))
                                    
    # ID 8 later for more players
    
