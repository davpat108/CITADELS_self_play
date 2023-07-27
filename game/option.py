from copy import deepcopy, copy
from game.deck import Deck, Card
from game.helper_classes import GameState, HandKnowledge, RolePropery, RoleKnowlage
from game.config import role_to_role_id

class option():
    def __init__(self, name, **kwargs):
        self.name = name
        self.attributes = kwargs

    def __eq__(self, other):
        return self.name == other.name and self.attributes == other.attributes

    # gamestates:	0 - choose role -> 1
	#1 2 gold or 2 cards -> 2 or 3
	#2 Which card to put back -> 3
	#3 Blackmail response -> 4 different character
	#4 Respond to response reveal -> 5 different character
	#5 Character ability/build/smithy/museum/lab/magic_school/weapon_storage -> next player 1 or 0 or end
	#6 graveyard -> 5 different character (warlord)
    #7 Magistrate reveal -> 5 different character
    #8 seer give back card

    # others
    def carry_out(self, game):
        winner = None
        

        if self.name == "role_pick":
            self.carry_out_role_pick(game)
        elif self.name == "gold_or_card":
            self.carry_out_gold_or_card(game)
        elif self.name == "which_card_to_keep":
            self.carry_out_put_back_card(game)
        elif self.name == "blackmail_response":
            self.carry_out_respond_to_blackmail(game)
        elif self.name == "reveal_blackmail_as_blackmailer":
            self.carry_out_responding_to_blackmail_response(game)
        elif self.name == "reveal_warrant_as_magistrate":
            self.carry_out_magistrate_reaveal(game)
        elif self.name == "build":
            self.carry_out_building(game)
        elif self.name == "empty_option":
            self.carry_out_empty(game)
        elif self.name == "finish_round":
            winner = self.finnish_main_sequnce_actions(game)

        elif self.name == "ghost_town_color_choice":
            self.carry_out_ghost_town(game)
        elif self.name == "smithy_choice":
            self.carry_out_smithy(game)
        elif self.name == "laboratory_choice":
            self.carry_out_laboratory(game)
        elif self.name == "magic_school_choice":
            self.carry_out_magic_school(game)
        elif self.name == "weapon_storage_choice":
            self.carry_out_weapon_storage(game)
        elif self.name == "lighthouse_choice":
            self.carry_out_lighthouse(game)
        elif self.name == "museum_choice":
            self.carry_out_museum(game)
        elif self.name == "graveyard":
            self.carry_out_graveyard(game)
        
        elif self.name == "take_gold_for_war":
            self.carry_out_take_gold_for_war(game)


        # roles
        # ID 0
        elif self.name == "assassination":
            self.carry_out_assasination(game)
        elif self.name == "magistrate_warrant":
            self.carry_out_warranting(game)
        elif self.name == "bewitching":
            self.carry_out_bewitching(game)

        #ID 1
        elif self.name == "steal":
            self.carry_out_stealing(game)
        elif self.name == "blackmail":
            self.carry_out_blackmail(game)
        elif self.name == "spy":
            self.carry_out_spying(game)

        #ID 2
        elif self.name == "magic_hand_change" or self.name == "discard_and_draw": # TODO make it 2 different options
            self.carry_out_magicking(game)
        elif self.name == "look_at_hand":
            self.carry_out_wizard_hand_looking(game)
        elif self.name == "take_from_hand":
            self.carry_out_wizard_take_from_hand(game)
        elif self.name == "seer":
            self.carry_out_seer_take_a_card(game)
        elif self.name == "give_back_card":
            self.carry_out_seer_give_back_cards(game)

        #ID 3
        elif self.name == "take_crown_king":
            self.carry_out_take_crown_king(game)
        elif self.name == "give_crown":
            self.carry_out_emperor(game)
        elif self.name == "take_crown_pat":
            self.carry_out_take_crown_patrician(game)

        #ID 4
        elif self.name == "bishop":
            self.carry_out_bishop(game)
        elif self.name == "cardinal_exchange":
            self.carry_out_cardinal(game)
        elif self.name == "abbot_gold_or_card":
            self.carry_out_abbot(game)
        elif self.name == "abbot_beg":
            self.carry_out_abbot_beg(game)

        #ID 5
        elif self.name == "merchant":
            self.carry_out_merchant(game)
        elif self.name == "alchemist":
            self.carry_out_alchemist(game)
        elif self.name == "trader":
            self.carry_out_trader(game)
        
        #ID 6
        elif self.name == "architect":
            self.carry_out_architect(game)
        elif self.name == "navigator_gold_card":
            self.carry_out_navigator(game)
        elif self.name == "scholar":
            self.carry_out_scholar_draw(game)
        elif self.name == "scholar_card_pick":
            self.carry_out_scholar_put_back(game)

        #ID 7
        elif self.name == "warlord_desctruction":
            self.carry_out_warlord(game)
        elif self.name == "marshal_steal":
            self.carry_out_marshal(game)
        elif self.name == "diplomat_exchange":
            self.carry_out_diplomat(game)

        #ID 8
        # Not yet
        game.check_game_ending(game.players[self.attributes['perpetrator']])

        return winner

    def carry_out_role_pick(self, game):
        game.players[self.attributes['perpetrator']].role = self.attributes['choice']
        game.roles_to_choose_from.pop(role_to_role_id[self.attributes['choice']])
        used_role = (role_to_role_id[self.attributes['choice']], self.attributes['choice'])

        for player in game.players:
            if player != game.players[self.attributes['perpetrator']]: 
                if game.turn_orders_for_roles.index(player.id) <  game.turn_orders_for_roles.index(game.players[self.attributes['perpetrator']].id):
                    possible_roles = list(game.roles.items())
                    if game.visible_face_up_role:
                        possible_roles.remove(list(game.visible_face_up_role.items())[0])
                    for role in game.roles_to_choose_from.items():
                        possible_roles.remove(role)
                    possible_roles.remove(used_role)
                    game.players[self.attributes['perpetrator']].known_roles[player.id].possible_roles=dict(sorted(possible_roles, key=lambda x: x[0]))

                else:
                    game.players[self.attributes['perpetrator']].known_roles[player.id].possible_roles=deepcopy(game.roles_to_choose_from)

        if game.players[self.attributes['perpetrator']].id != game.turn_orders_for_roles[-1]:
            game.gamestate.state = 0
            game.gamestate.player = game.players[game.turn_orders_for_roles[game.turn_orders_for_roles.index(game.players[self.attributes['perpetrator']].id) + 1]]
        else:
            refresh_used_roles(game)
            game.gamestate.state = 1
            game.gamestate.player = get_player_from_role_id(game.used_roles[0], game)


    def carry_out_gold_or_card(self, game):
        if game.role_properties[role_to_role_id[game.players[self.attributes['perpetrator']].role]].robbed:
            get_player_from_role_id(1, game).gold += game.players[self.attributes['perpetrator']].gold
            game.players[self.attributes['perpetrator']].gold = 0
            
        if self.attributes['choice'] == "gold":
            game.players[self.attributes['perpetrator']].gold += 2
            game.gamestate.state = 3
            game.gamestate.player = game.players[self.attributes['perpetrator']]
        else:
            # Astrology tower
            if Card(**{"suit":"unique", "type_ID":16, "cost": 5}) in game.players[self.attributes['perpetrator']].buildings.cards:
                for _ in range(3):
                    reshuffle_deck_if_empty(game)
                    game.players[self.attributes['perpetrator']].just_drawn_cards.add_card(game.deck.draw_card())
            else:
                for _ in range(2):
                    reshuffle_deck_if_empty(game)
                    game.players[self.attributes['perpetrator']].just_drawn_cards.add_card(game.deck.draw_card())
            game.gamestate.state = 2
            game.gamestate.player = game.players[self.attributes['perpetrator']]


    def carry_out_put_back_card(self, game):
        for card in self.attributes['choice']:
            game.players[self.attributes['perpetrator']].hand.add_card(game.players[self.attributes['perpetrator']].just_drawn_cards.get_a_card_like_it(card))
        for card in game.players[self.attributes['perpetrator']].just_drawn_cards.cards:
            game.deck.add_card(card)
        game.players[self.attributes['perpetrator']].just_drawn_cards.cards = []

        game.gamestate.state = 3
        game.gamestate.player = game.players[self.attributes['perpetrator']]

    def carry_out_empty(self, game):
        game.gamestate = self.attributes['next_gamestate']

    def carry_out_respond_to_blackmail(self, game):
        # Victims response
        if self.attributes['choice'] == "pay":
            get_player_from_role_id(1, game).gold += int(game.players[self.attributes['perpetrator']].gold/2)
            game.players[self.attributes['perpetrator']].gold -= int(game.players[self.attributes['perpetrator']].gold/2)
            game.gamestate.state = 5
            game.gamestate.player = game.players[self.attributes['perpetrator']]
        else:
            game.gamestate.state = 4
            game.gamestate.player = get_player_from_role_id(1, game)
            game.gamestate.next_gamestate = GameState(state=5, player=game.players[self.attributes['perpetrator']])


    def carry_out_responding_to_blackmail_response(self, game):
        # Its reversed as its the blackmailers response
        if self.attributes['choice'] == "reveal" and game.role_properties[role_to_role_id[game.players[self.attributes['target']].role]].blackmail == "Real":
            game.players[self.attributes['perpetrator']].gold += game.players[self.attributes['target']].gold
            game.players[self.attributes['target']].gold = 0
            for property in game.role_properties.values():
                property.blackmail = None
        game.gamestate = game.gamestate.next_gamestate

    def carry_out_magistrate_reaveal(self, game):
        if self.attributes['choice'] == "reveal" and game.role_properties[role_to_role_id[game.players[self.attributes['target']].role]].warrant == "Real":
            game.players[self.attributes['perpetrator']].buildings.add_card(game.players[self.attributes['target']].buildings.get_a_card_like_it(game.warrant_building))
            game.players[self.attributes['target']].gold += game.warrant_building.cost
            for property in game.role_properties.values():
                property.warrant = None
        game.gamestate = game.gamestate.next_gamestate

    def carry_out_building(self, game):
        game.players[self.attributes['perpetrator']].buildings.add_card(self.attributes['built_card'])
        if not game.players[self.attributes['perpetrator']].role == "Alchemist":
            game.players[self.attributes['perpetrator']].gold -= self.attributes['built_card'].cost

        if self.attributes['replica']:
            game.players[self.attributes['perpetrator']].replicas = self.attributes['replica']

        if self.attributes['built_card'].suit == "trade":
            game.gamestate.already_done_moves.append("trade_building")
        else:
            game.gamestate.already_done_moves.append("non_trade_building")
        
        if self.attributes['built_card'].type_ID == 29:
            game.players[self.attributes['perpetrator']].can_use_lighthouse = True

        # No warrant
        if game.role_properties[role_to_role_id[game.players[self.attributes['perpetrator']].role]].warrant is None:
            game.gamestate.state = 5
            game.gamestate.player = game.players[self.attributes['perpetrator']]
        # Warrant
        else:
            game.warrant_building = self.attributes['built_card']
            game.gamestate.state = 7
            game.gamestate.player = get_player_from_role_id(0, game)
            game.gamestate.next_gamestate = GameState(state=5, player=game.players[self.attributes['perpetrator']], already_done_moves=game.gamestate.already_done_moves)



    def carry_out_smithy(self, game):
        game.players[self.attributes['perpetrator']].gold -= 2
        for _ in range(3):
            reshuffle_deck_if_empty(game)
            game.players[self.attributes['perpetrator']].just_drawn_cards.add_card(game.deck.draw_card())
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append('smithy')
    
    def carry_out_laboratory(self, game):
        game.discard_deck.add_card(game.players[self.attributes['perpetrator']].hand.get_a_card_like_it(self.attributes['choice']))
        game.players[self.attributes['perpetrator']].gold += 1
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append('lab')

    def carry_out_magic_school(self, game):
        current_magic_school_suit = [card.suit for card in game.players[self.attributes['perpetrator']].buildings.cards if card.type_ID == 25]
        game.players[self.attributes['perpetrator']].buildings.get_a_card_like_it(Card(**{"suit":current_magic_school_suit, "type_ID":25, "cost": 6}))
        game.players[self.attributes['perpetrator']].buildings.add_card(Card(**{"suit":self.attributes['choice'], "type_ID":25, "cost": 6}))
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append('magic_school')

    def carry_out_ghost_town(self, game):
        game.players[self.attributes['perpetrator']].buildings.get_a_card_like_it(Card(**{"suit":"unique", "type_ID":19, "cost": 2}))
        game.players[self.attributes['perpetrator']].buildings.add_card(Card(**{"suit":self.attributes['choice'], "type_ID":19, "cost": 2}))
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
    
    def carry_out_museum(self, game):
        game.players[self.attributes['perpetrator']].museum_cards.add_card(game.players[self.attributes['perpetrator']].hand.get_a_card_like_it(self.attributes['choice']))
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append('museum')

    def carry_out_weapon_storage(self, game):
        game.discard_deck.add_card(game.players[self.attributes['perpetrator']].buildings.get_a_card_like_it(Card(**{"suit":"unique", "type_ID":27, "cost": 3})))
        game.discard_deck.add_card(game.players[self.attributes['target']].buildings.get_a_card_like_it(self.attributes['choice']))
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]

    def carry_out_lighthouse(self, game):
        game.players[self.attributes['perpetrator']].known_hands.append(HandKnowledge(player_id=-1, hand=deepcopy(game.deck), confidence=5))
        game.players[self.attributes['perpetrator']].hand.add_card(game.deck.get_a_card_like_it(self.attributes['choice']))
        game.players[self.attributes['perpetrator']].can_use_lighthouse = False
        game.deck.shuffle_deck()

        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]


    def carry_out_graveyard(self, game):
        # The destroyed building is appended to the discard deck so pop(-1) gets it
        game.players[self.attributes['perpetrator']].buildings.add_card(game.discard_deck.cards.pop(-1))
        game.players[self.attributes['perpetrator']].gold -= 1
        game.gamestate = game.gamestate.next_gamestate

    def finnish_main_sequnce_actions(self, game):
        # Deciding that I did enough in my turn
        # Park
        if not game.role_properties[role_to_role_id[game.players[self.attributes['perpetrator']].role]].dead:
            if Card(**{"suit":"unique", "type_ID":28, "cost": 6}) in game.players[self.attributes['perpetrator']].buildings.cards:
                if len(game.players[self.attributes['perpetrator']].hand.cards) == 0:
                    for _ in range(2):
                        reshuffle_deck_if_empty(game)
                        game.players[self.attributes['perpetrator']].just_drawn_cards.add_card(game.deck.draw_card())            
            # Poorhouse    
            if Card(**{"suit":"unique", "type_ID":30, "cost": 5}) in game.players[self.attributes['perpetrator']].buildings.cards:
                if len(game.players[self.attributes['perpetrator']].hand.cards) == 0:
                    game.players[self.attributes['perpetrator']].gold += 1
        # witch
        if self.attributes['next_witch']:
            game.gamestate.state = 5
            game.gamestate.player = get_player_from_role_id(game.used_roles[0], game)
            game.gamestate.player.role = game.players[self.attributes['perpetrator']].role
            game.role_properties[role_to_role_id[game.players[self.attributes['perpetrator']].role]].possessed = False
            game.gamestate.already_done_moves = []
            return False
        
        # I was the last player in the round
        if game.used_roles[-1] == role_to_role_id[game.players[self.attributes['perpetrator']].role]:
            if game.ending:
                points = []
                for player in game.players:
                    points.append(player.count_points())
                game.points = points
                game.terminal = True
                return game.players[points.index(max(points))]
            game.setup_round()

        # Not the last player
        else:
            game.gamestate.state = 1
            game.gamestate.player = get_player_from_role_id(game.used_roles[game.used_roles.index(role_to_role_id[game.players[self.attributes['perpetrator']].role])+1], game)
            game.gamestate.already_done_moves = []

        return False
    # ID 0
    def carry_out_assasination(self, game):
        game.role_properties[self.attributes['target']].dead = True
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
        
    def carry_out_warranting(self, game):
        game.role_properties[self.attributes['real_target']].warrant = "Real"
        game.role_properties[self.attributes['fake_targets'][0]].warrant = "Fake"
        game.role_properties[self.attributes['fake_targets'][1]].warrant = "Fake"
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_bewitching(self, game):
        game.role_properties[self.attributes['target']].possessed = True
        game.gamestate.state = 1
        game.gamestate.player = get_player_from_role_id(game.used_roles[game.used_roles.index(role_to_role_id[game.players[self.attributes['perpetrator']].role])+1], game)
        
    # ID 1
    def carry_out_stealing(self, game):
        # TODO this is wrong like this
        game.role_properties[self.attributes['target']].robbed = True
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
    
    def carry_out_blackmail(self, game):
        game.role_properties[self.attributes['real_target']].blackmail = "Real"
        game.role_properties[self.attributes['fake_target']].blackmail = "Fake"
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
        
    def carry_out_spying(self, game):
        cards_to_draw = sum([1 if building.suit == self.attributes['suit'] else 0 for building in game.players[self.attributes['target']].hand.cards])
        gold_to_steal = min(cards_to_draw, game.players[self.attributes['target']].gold)

        game.players[self.attributes['perpetrator']].gold += gold_to_steal
        game.players[self.attributes['target']].gold -= gold_to_steal

        reshuffle_deck_if_empty(game)
        game.players[self.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    # ID 2
    def carry_out_magicking(self, game):
        if self.name == "magic_hand_change":
            game.players[self.attributes['perpetrator']].hand.cards, game.players[self.attributes['target']].hand.cards = game.players[self.attributes['target']].hand.cards, game.players[self.attributes['perpetrator']].hand.cards

        if self.name == "discard_and_draw":
            for card in game.players[self.attributes['perpetrator']].hand.cards:
                game.deck.add_card(game.players[self.attributes['perpetrator']].hand.get_a_card_like_it(card))
            for _ in range(len(game.players[self.attributes['perpetrator']].hand.cards)):
                reshuffle_deck_if_empty(game)
                game.players[self.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_wizard_hand_looking(self, game):
        game.players[self.attributes['perpetrator']].known_hands.append(HandKnowledge(player_id=self.attributes['target'], hand=deepcopy(game.players[self.attributes['target']].hand), confidence=5, wizard=True))
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_wizard_take_from_hand(self, game):
        # Can be the same building you already have, unlike with regualr building
        if self.attributes['build']:
            wizard_hand_knowlage = next((hand_knowlage for hand_knowlage in game.players[self.attributes['perpetrator']].known_hands if hand_knowlage.wizard), None)
            self.carry_out_building(game)
            # It has to be the last one in this position
            wizard_hand_knowlage.hand.get_a_card_like_it(self.attributes['built_card'])
        else:
            wizard_hand_knowlage = next((hand_knowlage for hand_knowlage in game.players[self.attributes['perpetrator']].known_hands if hand_knowlage.wizard), None)

            game.players[self.attributes['perpetrator']].hand.add_card(game.players[self.attributes['target']].hand.get_a_card_like_it(self.attributes['card']))
            # It has to be the last one in this position
            wizard_hand_knowlage.hand.get_a_card_like_it(self.attributes['card'])
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("took_from_hand")

    def carry_out_seer_take_a_card(self, game):
        game.seer_taken_card_from = []
        for player in game.players:
            if player.id != game.players[self.attributes['perpetrator']].id and player.hand.cards:
                player.hand.shuffle_deck()
                reshuffle_deck_if_empty(game)
                game.players[self.attributes['perpetrator']].hand.add_card(player.hand.draw_card())
                game.seer_taken_card_from.append(player.id)
        game.gamestate.state = 8
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
        game.gamestate.next_gamestate = GameState(state=5, player=game.players[self.attributes['perpetrator']], already_done_moves=game.gamestate.already_done_moves)

    def carry_out_seer_give_back_cards(self, game):
        for handout in self.attributes['card_handouts'].items():
            added_deck = Deck(empty=True)
            added_deck.add_card(handout[1])
            game.players[handout[0]].hand.add_card(game.players[self.attributes['perpetrator']].hand.get_a_card_like_it(handout[1]))
            game.players[self.attributes['perpetrator']].known_hands.append(HandKnowledge(player_id=handout[0], hand=added_deck, confidence=5))
        game.seer_taken_card_from = []
        game.gamestate = game.gamestate.next_gamestate


    # ID 3
    def carry_out_take_crown_king(self, game):
        for player in game.players:
            if player.crown:
                player.crown = False
                break
        for building in game.players[self.attributes['perpetrator']].buildings.cards:
            if building.suit == "lord":
                game.players[self.attributes['perpetrator']].gold += 1

        game.players[self.attributes['perpetrator']].crown = True
        troneroom_owner_gold(game)
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_take_crown_patrician(self, game):
        for player in game.players:
            if player.crown:
                player.crown = False
                break
        for building in game.players[self.attributes['perpetrator']].buildings.cards:
            if building.suit == "lord":
                reshuffle_deck_if_empty(game)
                game.players[self.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
                
        game.players[self.attributes['perpetrator']].crown = True
        troneroom_owner_gold(game)
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_emperor(self, game):
        for building in game.players[self.attributes['perpetrator']].buildings.cards:
            if building.suit == "lord":
                game.players[self.attributes['perpetrator']].gold += 1
        
        if self.attributes['gold_or_card'] == "card":
            game.players[self.attributes['target']].hand.shuffle_deck()
            game.players[self.attributes['perpetrator']].hand.add_card(game.players[self.attributes['target']].hand.draw_card())

        if self.attributes['gold_or_card'] == "gold":
            game.players[self.attributes['perpetrator']].gold += 1
            game.players[self.attributes['target']].gold -= 1

        if not game.role_properties[3].possessed:
            game.players[self.attributes['perpetrator']].crown = False
            game.players[self.attributes['target']].crown = True
            troneroom_owner_gold(game)
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
            

    # ID 4
    def carry_out_bishop(self, game):
        for building in game.players[self.attributes['perpetrator']].buildings.cards:
            if building.suit == "religion":
                game.players[self.attributes['perpetrator']].gold += 1
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
    
    def carry_out_abbot(self, game):
        game.players[self.attributes['perpetrator']].gold += self.attributes['gold_or_card_combination'].count("gold")
        for _ in range(self.attributes['gold_or_card_combination'].count("card")):
            reshuffle_deck_if_empty(game)
            game.players[self.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
                
    def carry_out_abbot_beg(self, game):
        max_gold_player_index = max(enumerate([player.gold for player in game.players]), key=lambda x: x[1])[0]
        game.players[max_gold_player_index].gold -= 1
        get_player_from_role_id(4, game).gold += 1
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("begged")
            
    def carry_out_cardinal(self, game):
        # Special way of building


        # Regular building
        game.players[self.attributes['perpetrator']].buildings.add_card(game.players[self.attributes['perpetrator']].hand.get_a_card_like_it(self.attributes['built_card']))
        game.players[self.attributes['perpetrator']].gold -= self.attributes['built_card'].cost-self.attributes['factory']
        game.players[self.attributes['perpetrator']].gold = max(0, game.players[self.attributes['perpetrator']].gold)
        if self.attributes['replica']:
            game.players[self.attributes['perpetrator']].replicas = self.attributes['replica']

        # Cardinal card take
        if self.attributes['cards_to_give']:
            game.players[self.attributes['target']].gold -= len(self.attributes['cards_to_give'])
            for card in self.attributes['cards_to_give']:
                game.players[self.attributes['target']].hand.add_card(game.players[self.attributes['perpetrator']].hand.get_a_card_like_it(card))
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    # ID 5
    def carry_out_merchant(self, game):
        for building in game.players[self.attributes['perpetrator']].buildings.cards:
            if building.suit == "trade":
                game.players[self.attributes['perpetrator']].gold += 1
        game.players[self.attributes['perpetrator']].gold += 1
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
    
    def carry_out_alchemist(self, game):
        pass
        # Cost is zero

    def carry_out_trader(self, game):
        for building in game.players[self.attributes['perpetrator']].buildings.cards:
            if building.suit == "religion":
                game.players[self.attributes['perpetrator']].gold += 1
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    # ID 6
    def carry_out_architect(self, game):
        # Building Limit is three
        for _ in range(2):
            reshuffle_deck_if_empty(game)
            game.players[self.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_navigator(self, game):
        # Building Limit is zero
        if self.attributes['choice'] == "4card":
            for _ in range(4):
                reshuffle_deck_if_empty(game)
                game.players[self.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
        if self.attributes['choice'] == "4gold":
            game.players[self.attributes['perpetrator']].gold += 4
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_scholar_draw(self, game):
        game.seven_drawn_cards = Deck(empty=True)
        for _ in range(min(7, len(game.deck.cards))):
            reshuffle_deck_if_empty(game)
            card = game.deck.draw_card()
            game.players[self.attributes['perpetrator']].hand.add_card(card)
            game.seven_drawn_cards.add_card(card)

        game.gamestate.state = 9
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
        game.gamestate.next_gamestate = GameState(state=5, player=game.players[self.attributes['perpetrator']], already_done_moves=game.gamestate.already_done_moves)

    def carry_out_scholar_put_back(self, game):
        for card in self.attributes['unchosen_cards'].cards:
            game.deck.add_card(game.players[self.attributes['perpetrator']].hand.get_a_card_like_it(card))
        game.gamestate = game.gamestate.next_gamestate
        game.seven_drawn_cards = []

    # ID 7
    def carry_out_marshal(self, game):
        game.players[self.attributes['perpetrator']].gold -= self.attributes['choice'].cost
        game.players[self.attributes['target']].gold += self.attributes['choice'].cost
        game.players[self.attributes['perpetrator']].buildings.add_card(game.players[self.attributes['target']].buildings.get_a_card_like_it(self.attributes['choice']))
        if check_if_building_is_replica(game.players[self.attributes['target']], self.attributes['choice']):
            game.players[self.attributes['target']].replicas -= 1
        settle_museum(self, game)
        settle_lighthouse(self, game)
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_warlord(self, game):
        game.players[self.attributes['perpetrator']].gold -= self.attributes['choice'].cost-1
        game.discard_deck.add_card(game.players[self.attributes['target']].buildings.get_a_card_like_it(self.attributes['choice']))

        if check_if_building_is_replica(game.players[self.attributes['target']], self.attributes['choice']):
            game.players[self.attributes['target']].replicas -= 1
            
        settle_museum(self, game)
        settle_lighthouse(self, game)
            
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")
        graveyard_owner = get_graveyard_owner(game)
        if graveyard_owner is not None and graveyard_owner != game.players[self.attributes['perpetrator']]:
            game.gamestate.state = 6
            game.gamestate.player = graveyard_owner
            game.gamestate.next_gamestate = GameState(state=5, player=game.players[self.attributes['perpetrator']], already_done_moves=["character_ability"])


    def carry_out_diplomat(self, game):
        game.players[self.attributes['perpetrator']].gold -= self.attributes['money_owed']
        game.players[self.attributes['target']].gold += self.attributes['money_owed']
        game.players[self.attributes['perpetrator']].buildings.add_card(game.players[self.attributes['target']].buildings.get_a_card_like_it(self.attributes['choice']))
        game.players[self.attributes['target']].buildings.add_card(game.players[self.attributes['perpetrator']].buildings.get_a_card_like_it(self.attributes['give']))

        if check_if_building_is_replica(game.players[self.attributes['target']], self.attributes['choice']):
            game.players[self.attributes['target']].replicas -= 1
        settle_museum(self, game)
        settle_lighthouse(self, game)


        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("character_ability")

    def carry_out_take_gold_for_war(self, game):
        for building in game.players[self.attributes['perpetrator']].buildings.cards:
            if building.suit == "war":
                game.players[self.attributes['perpetrator']].gold += 1
        game.gamestate.state = 5
        game.gamestate.player = game.players[self.attributes['perpetrator']]
        game.gamestate.already_done_moves.append("take_gold")



def reshuffle_deck_if_empty(game):
    if not len(game.deck.cards):
        if not len(game.discard_deck.cards):
            return
        game.discard_deck.shuffle_deck()
        game.deck = deepcopy(game.discard_deck)
        game.discard_deck = Deck(empty=True)

def get_player_from_role_id(role_id, game):
    for player in game.players:
        if player.role == game.roles[role_id]:
            return player
    return None

def settle_lighthouse(option, game):
    if option.attributes['choice'] == Card(**{"suit":"unique", "type_ID":29, "cost": 3}) and game.players[option.attributes['target']].can_use_lighthouse:
        game.players[option.attributes['target']].can_use_lighthouse = False
        game.players[option.attributes['perpetrator']].can_use_lighthouse = True

def settle_museum(option, game):
    if option.name == "warlord_desctruction":
        if option.attributes['choice'] == Card(**{"suit":"unique", "type_ID":34, "cost": 4}):
            for _ in range(len(game.players[option.attributes['target']].museum_cards.cards)):
                game.discard_deck.add_card(game.players[option.attributes['target']].museum_cards.draw_card())
    else:
        if option.attributes['choice'] == Card(**{"suit":"unique", "type_ID":34, "cost": 4}):
            for _ in range(len(game.players[option.attributes['target']].museum_cards.cards)):
                game.players[option.attributes['perpetrator']].museum_cards.add_card(game.players[option.attributes['target']].museum_cards.draw_card())

def troneroom_owner_gold(game):
    trone_room_owner = None
    for player in game.players:
        if Card(**{"suit":"unique", "type_ID":32, "cost": 6}) in player.buildings.cards:
            trone_room_owner = player
            break
    if trone_room_owner:
        trone_room_owner.gold += 1

def get_graveyard_owner(game):
    for player in game.players:
        if Card(**{"suit":"unique", "type_ID":24, "cost": 5}) in player.buildings.cards:
            return player
    return None

def check_if_building_is_replica(target_player, building):
    if building in target_player.buildings.cards and target_player.buildings.cards.count(building) > 1:
        return True
    return False


def refresh_used_roles(game):
    for player in game.players:
        game.used_roles.append(role_to_role_id[player.role])
    game.used_roles.sort()
