from copy import deepcopy
from game.deck import Deck, Card
from game.helper_classes import GameState, HandKnowledge
from game.config import role_to_role_id

def carry_out_role_pick(option, game):
    game.players[option.attributes['perpetrator']].role = option.attributes['choice']
    game.roles_to_choose_from.pop(role_to_role_id[option.attributes['choice']])
    used_role = (role_to_role_id[option.attributes['choice']], option.attributes['choice'])
    
    # Settling role knowledge of other players
    for player in game.players:
        if player != game.players[option.attributes['perpetrator']]:
            # Before this player
            if game.turn_orders_for_roles.index(player.id) <  game.turn_orders_for_roles.index(option.attributes['perpetrator']):
                possible_roles = list(game.roles.items())
                if game.visible_face_up_role:
                    possible_roles.remove(list(game.visible_face_up_role.items())[0])
                for role in game.roles_to_choose_from.items():
                    possible_roles.remove(role)
                possible_roles.remove(used_role)
                game.players[option.attributes['perpetrator']].known_roles[player.id].possible_roles=dict(sorted(possible_roles, key=lambda x: x[0]))
            # After this player
            else:
                game.players[option.attributes['perpetrator']].known_roles[player.id].possible_roles=deepcopy(game.roles_to_choose_from)
    if option.attributes['perpetrator'] != game.turn_orders_for_roles[-1]:
        game.gamestate.state = 0
        game.gamestate.player_id = game.turn_orders_for_roles[game.turn_orders_for_roles.index(option.attributes['perpetrator']) + 1]
    else:
        game.setup_next_player()


def carry_out_gold_or_card(option, game):
    # Reveal role
    confirm_role_knowledges(game.players[option.attributes['perpetrator']], game)
    
    if game.role_properties[role_to_role_id[game.players[option.attributes['perpetrator']].role]].robbed:
        game.get_player_from_role_id(1).gold += game.players[option.attributes['perpetrator']].gold
        game.players[option.attributes['perpetrator']].gold = 0
    if option.attributes['choice'] == "gold":
        game.players[option.attributes['perpetrator']].gold += 2
        game.gamestate.state = 3
        game.gamestate.player_id = option.attributes['perpetrator']
    else:
        # Astrology tower
        if Card(**{"suit":"unique", "type_ID":16, "cost": 5}) in game.players[option.attributes['perpetrator']].buildings.cards:
            for _ in range(3):
                reshuffle_deck_if_empty(game)
                game.players[option.attributes['perpetrator']].just_drawn_cards.add_card(game.deck.draw_card())
        else:
            for _ in range(2):
                reshuffle_deck_if_empty(game)
                game.players[option.attributes['perpetrator']].just_drawn_cards.add_card(game.deck.draw_card())
        game.gamestate.state = 2
        game.gamestate.player_id = option.attributes['perpetrator']


def carry_out_put_back_card(option, game):
    for card in option.attributes['choice']:
        game.players[option.attributes['perpetrator']].hand.add_card(game.players[option.attributes['perpetrator']].just_drawn_cards.get_a_card_like_it(card))
    for card in game.players[option.attributes['perpetrator']].just_drawn_cards.cards:
        game.deck.add_card(card)
    game.players[option.attributes['perpetrator']].just_drawn_cards.cards = []
    
    game.gamestate.state = 3
    game.gamestate.player_id = option.attributes['perpetrator']

def carry_out_empty(option, game):
    game.gamestate = option.attributes['next_gamestate']

def carry_out_respond_to_blackmail(option, game):
    # Victims response
    if option.attributes['choice'] == "pay":
        game.get_player_from_role_id(1).gold += int(game.players[option.attributes['perpetrator']].gold/2)
        game.players[option.attributes['perpetrator']].gold -= int(game.players[option.attributes['perpetrator']].gold/2)
        game.gamestate.state = 5
        game.gamestate.player_id = option.attributes['perpetrator']
    else:
        game.gamestate.state = 4
        game.gamestate.player_id = game.get_player_from_role_id(1).id
        game.gamestate.interruption = True
        game.gamestate.next_gamestate = GameState(state=5, player_id=option.attributes['perpetrator'])


def carry_out_responding_to_blackmail_response(option, game):
    # Its reversed as its the blackmailers response
    if option.attributes['choice'] == "reveal" and game.role_properties[role_to_role_id[game.players[option.attributes['target']].role]].blackmail == "Real":
        game.players[option.attributes['perpetrator']].gold += game.players[option.attributes['target']].gold
        game.players[option.attributes['target']].gold = 0
        for property in game.role_properties.values():
            property.blackmail = None
    game.gamestate = game.gamestate.next_gamestate
    
def carry_out_magistrate_reaveal(option, game):
    if option.attributes['choice'] == "reveal" and game.role_properties[role_to_role_id[game.players[option.attributes['target']].role]].warrant == "Real":
        game.players[option.attributes['perpetrator']].buildings.add_card(game.players[option.attributes['target']].buildings.get_a_card_like_it(game.warrant_building))
        game.players[option.attributes['target']].gold += game.warrant_building.cost
        for property in game.role_properties.values():
            property.warrant = None
    game.gamestate = game.gamestate.next_gamestate

def carry_out_building(option, game):
    game.players[option.attributes['perpetrator']].buildings.add_card(game.players[option.attributes['perpetrator']].hand.get_a_card_like_it(option.attributes['built_card']))
    if not game.players[option.attributes['perpetrator']].role == "Alchemist":
        game.players[option.attributes['perpetrator']].gold -= option.attributes['built_card'].cost
        
    if option.attributes['replica']:
        game.players[option.attributes['perpetrator']].replicas = option.attributes['replica']
        
    if option.attributes['built_card'].suit == "trade":
        game.gamestate.already_done_moves.append("trade_building")
    else:
        game.gamestate.already_done_moves.append("non_trade_building")
    
    if option.attributes['built_card'].type_ID == 29:
        game.players[option.attributes['perpetrator']].can_use_lighthouse = True
    # No warrant
    if game.role_properties[role_to_role_id[game.players[option.attributes['perpetrator']].role]].warrant is None:
        game.gamestate.state = 5
        game.gamestate.player_id = option.attributes['perpetrator']
    # Warrant
    else:
        game.warrant_building = option.attributes['built_card']
        game.gamestate.state = 7
        game.gamestate.player_id = game.get_player_from_role_id(0).id
        game.gamestate.interruption = True
        game.gamestate.next_gamestate = GameState(state=5, player_id=option.attributes['perpetrator'], already_done_moves=game.gamestate.already_done_moves)



def carry_out_smithy(option, game):
    game.players[option.attributes['perpetrator']].gold -= 2
    for _ in range(3):
        reshuffle_deck_if_empty(game)
        game.players[option.attributes['perpetrator']].just_drawn_cards.add_card(game.deck.draw_card())
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append('smithy')
    
def carry_out_laboratory(option, game):
    game.discard_deck.add_card(game.players[option.attributes['perpetrator']].hand.get_a_card_like_it(option.attributes['choice']))
    game.players[option.attributes['perpetrator']].gold += 1
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append('lab')

def carry_out_magic_school(option, game):
    current_magic_school_suit = [card.suit for card in game.players[option.attributes['perpetrator']].buildings.cards if card.type_ID == 25]
    game.players[option.attributes['perpetrator']].buildings.get_a_card_like_it(Card(**{"suit":current_magic_school_suit, "type_ID":25, "cost": 6}))
    game.players[option.attributes['perpetrator']].buildings.add_card(Card(**{"suit":option.attributes['choice'], "type_ID":25, "cost": 6}))
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append('magic_school')

def carry_out_ghost_town(option, game):
    game.players[option.attributes['perpetrator']].buildings.get_a_card_like_it(Card(**{"suit":"unique", "type_ID":19, "cost": 2}))
    game.players[option.attributes['perpetrator']].buildings.add_card(Card(**{"suit":option.attributes['choice'], "type_ID":19, "cost": 2}))
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    
def carry_out_museum(option, game):
    game.players[option.attributes['perpetrator']].museum_cards.add_card(game.players[option.attributes['perpetrator']].hand.get_a_card_like_it(option.attributes['choice']))
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append('museum')

def carry_out_weapon_storage(option, game):
    game.discard_deck.add_card(game.players[option.attributes['perpetrator']].buildings.get_a_card_like_it(Card(**{"suit":"unique", "type_ID":27, "cost": 3})))
    game.discard_deck.add_card(game.players[option.attributes['target']].buildings.get_a_card_like_it(option.attributes['choice']))
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']

def carry_out_lighthouse(option, game):
    game.players[option.attributes['perpetrator']].known_hands.append(HandKnowledge(player_id=-1, hand=deepcopy(game.deck), confidence=5))
    game.players[option.attributes['perpetrator']].hand.add_card(game.deck.get_a_card_like_it(option.attributes['choice']))
    game.players[option.attributes['perpetrator']].can_use_lighthouse = False
    game.deck.shuffle_deck()
    
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']


def carry_out_graveyard(option, game):
    # The destroyed building is appended to the discard deck so pop(-1) gets it
    game.players[option.attributes['perpetrator']].buildings.add_card(game.discard_deck.cards.pop(-1))
    game.players[option.attributes['perpetrator']].gold -= 1
    game.gamestate = game.gamestate.next_gamestate

def finish_main_sequnce_actions(option, game):
    # Deciding that I did enough in my turn
    # Park
    if not game.role_properties[role_to_role_id[game.players[option.attributes['perpetrator']].role]].dead:
        if Card(**{"suit":"unique", "type_ID":28, "cost": 6}) in game.players[option.attributes['perpetrator']].buildings.cards:
            if len(game.players[option.attributes['perpetrator']].hand.cards) == 0:
                for _ in range(2):
                    reshuffle_deck_if_empty(game)
                    game.players[option.attributes['perpetrator']].just_drawn_cards.add_card(game.deck.draw_card())            
        # Poorhouse    
        if Card(**{"suit":"unique", "type_ID":30, "cost": 5}) in game.players[option.attributes['perpetrator']].buildings.cards:
            if len(game.players[option.attributes['perpetrator']].hand.cards) == 0:
                game.players[option.attributes['perpetrator']].gold += 1
                
    if option.attributes['crown']:
        confirm_role_knowledges(game.players[option.attributes['perpetrator']], game)
        move_crown(game, option.attributes['perpetrator'])
        
    # witch
    elif game.role_properties[role_to_role_id[game.players[option.attributes['perpetrator']].role]].dead:
        confirm_role_knowledges(game.players[option.attributes['perpetrator']], game)
        
    if option.attributes['next_witch']:
        # Witch is making the choices
        game.gamestate.state = 5
        game.gamestate.player_id = game.get_player_from_role_id(0).id
        # Witch takes over role
        game.players[game.gamestate.player_id].role = game.players[option.attributes['perpetrator']].role
        game.role_properties[role_to_role_id[game.players[option.attributes['perpetrator']].role]].possessed = False
        # Bewitched role to avoid 2 people with the same role
        game.players[option.attributes['perpetrator']].role = "Bewitched"
        
        # Update known roles with witches new role
        for player in game.players:
            if player.id != game.gamestate.player_id:
                player.known_roles[game.gamestate.player_id].possible_roles = {role_to_role_id[game.players[game.gamestate.player_id].role] : game.players[game.gamestate.player_id].role}
            if player.id != option.attributes['perpetrator']:
                player.known_roles[option.attributes['perpetrator']].possible_roles = {-1 : game.players[option.attributes['perpetrator']].role}
        
        game.gamestate.already_done_moves = []
        
        return False
    
    # I was the last player in the round
    if game.used_roles[-1] == role_to_role_id[game.players[option.attributes['perpetrator']].role]:
        winner = game.check_game_ending()
        if not winner:
            game.setup_round()
        else:
            return winner
        
    # Not the last player
    else:
        game.setup_next_player(current_player_id=option.attributes['perpetrator'])
    return False
    # ID 0
def carry_out_assasination(option, game):
    game.role_properties[option.attributes['choice']].dead = True
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
        
def carry_out_warranting(option, game):
    game.role_properties[option.attributes['real_target']].warrant = "Real"
    game.role_properties[option.attributes['fake_targets'][0]].warrant = "Fake"
    game.role_properties[option.attributes['fake_targets'][1]].warrant = "Fake"
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

def carry_out_bewitching(option, game):
    game.role_properties[option.attributes['choice']].possessed = True
    game.players[option.attributes['perpetrator']].witch = True
    game.setup_next_player(current_player_id=option.attributes['perpetrator'])
        
    # ID 1
def carry_out_stealing(option, game):
    game.role_properties[option.attributes['choice']].robbed = True
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
    
def carry_out_blackmail(option, game):
    game.role_properties[option.attributes['real_target']].blackmail = "Real"
    game.role_properties[option.attributes['fake_target']].blackmail = "Fake"
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
        
def carry_out_spying(option, game):
    cards_to_draw = sum([1 if building.suit == option.attributes['suit'] else 0 for building in game.players[option.attributes['target']].hand.cards])
    gold_to_steal = min(cards_to_draw, game.players[option.attributes['target']].gold)
    game.players[option.attributes['perpetrator']].gold += gold_to_steal
    game.players[option.attributes['target']].gold -= gold_to_steal
    
    reshuffle_deck_if_empty(game)
    game.players[option.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

    # ID 2
def carry_out_magicking(option, game):
    if option.name == "magic_hand_change":
        game.players[option.attributes['perpetrator']].hand.cards, game.players[option.attributes['target']].hand.cards = game.players[option.attributes['target']].hand.cards, game.players[option.attributes['perpetrator']].hand.cards
    
    if option.name == "discard_and_draw":
        for card in game.players[option.attributes['perpetrator']].hand.cards:
            game.deck.add_card(game.players[option.attributes['perpetrator']].hand.get_a_card_like_it(card))
        for _ in range(len(game.players[option.attributes['perpetrator']].hand.cards)):
            reshuffle_deck_if_empty(game)
            game.players[option.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

def carry_out_wizard_hand_looking(option, game):
    game.players[option.attributes['perpetrator']].known_hands.append(HandKnowledge(player_id=option.attributes['target'], hand=deepcopy(game.players[option.attributes['target']].hand), confidence=5, wizard=True))
    game.gamestate.state = 10
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
    game.gamestate.next_gamestate = GameState(state=5, player_id=option.attributes['perpetrator'], already_done_moves=game.gamestate.already_done_moves)

def carry_out_wizard_take_from_hand(option, game):
    # Can be the same building you already have, unlike with regualr building
    if option.attributes['build']:
        wizard_hand_knowlage = next((hand_knowlage for hand_knowlage in game.players[option.attributes['perpetrator']].known_hands if hand_knowlage.wizard), None)
        game.players[option.attributes['perpetrator']].hand.add_card(game.players[option.attributes['target']].hand.get_a_card_like_it(option.attributes['built_card']))
        option.attributes['replica'] = game.players[option.attributes['perpetrator']].buildings.cards.count(option.attributes['built_card'])
        carry_out_building(option, game)
        # It has to be the last one in this position
        wizard_hand_knowlage.hand.get_a_card_like_it(option.attributes['built_card'])
        
    else:
        wizard_hand_knowlage = next((hand_knowlage for hand_knowlage in game.players[option.attributes['perpetrator']].known_hands if hand_knowlage.wizard), None)
        game.players[option.attributes['perpetrator']].hand.add_card(game.players[option.attributes['target']].hand.get_a_card_like_it(option.attributes['card']))
        # It has to be the last one in this position
        wizard_hand_knowlage.hand.get_a_card_like_it(option.attributes['card'])
        
    game.gamestate = game.gamestate.next_gamestate

def carry_out_seer_take_a_card(option, game):
    game.seer_taken_card_from = []
    for player in game.players:
        if player.id != game.players[option.attributes['perpetrator']].id and player.hand.cards:
            player.hand.shuffle_deck()
            reshuffle_deck_if_empty(game)
            game.players[option.attributes['perpetrator']].hand.add_card(player.hand.draw_card())
            game.seer_taken_card_from.append(player.id)
    game.gamestate.state = 8
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
    game.gamestate.next_gamestate = GameState(state=5, player_id=option.attributes['perpetrator'], already_done_moves=game.gamestate.already_done_moves)

def carry_out_seer_give_back_cards(option, game):
    for handout in option.attributes['card_handouts'].items():
        added_deck = Deck(empty=True)
        added_deck.add_card(handout[1])
        game.players[handout[0]].hand.add_card(game.players[option.attributes['perpetrator']].hand.get_a_card_like_it(handout[1]))
        game.players[option.attributes['perpetrator']].known_hands.append(HandKnowledge(player_id=handout[0], hand=deepcopy(added_deck), confidence=5))
    game.seer_taken_card_from = []
    game.gamestate = game.gamestate.next_gamestate


# ID 3
def carry_out_take_crown_king(option, game):
    for building in game.players[option.attributes['perpetrator']].buildings.cards:
        if building.suit == "lord":
            game.players[option.attributes['perpetrator']].gold += 1
            
    if not game.players[option.attributes['perpetrator']].witch:
        move_crown(game, option.attributes['perpetrator'])
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

def carry_out_take_crown_patrician(option, game):
    for building in game.players[option.attributes['perpetrator']].buildings.cards:
        if building.suit == "lord":
            reshuffle_deck_if_empty(game)
            game.players[option.attributes['perpetrator']].hand.add_card(game.deck.draw_card())

    if not game.players[option.attributes['perpetrator']].witch:
        move_crown(game, option.attributes['perpetrator'])
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

def carry_out_emperor(option, game):
    for building in game.players[option.attributes['perpetrator']].buildings.cards:
        if building.suit == "lord":
            game.players[option.attributes['perpetrator']].gold += 1
    
    if option.attributes['gold_or_card'] == "card":
        game.players[option.attributes['target']].hand.shuffle_deck()
        game.players[option.attributes['perpetrator']].hand.add_card(game.players[option.attributes['target']].hand.draw_card())
        
    if option.attributes['gold_or_card'] == "gold":
        game.players[option.attributes['perpetrator']].gold += 1
        game.players[option.attributes['target']].gold -= 1
    confirm_role_knowledges(game.players[option.attributes['perpetrator']], game)
    move_crown(game, option.attributes['target'])
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
            

# ID 4
def carry_out_bishop(option, game):
    for building in game.players[option.attributes['perpetrator']].buildings.cards:
        if building.suit == "religion":
            game.players[option.attributes['perpetrator']].gold += 1
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
    
def carry_out_abbot(option, game):
    game.players[option.attributes['perpetrator']].gold += option.attributes['gold_or_card_combination'].count("gold")
    for _ in range(option.attributes['gold_or_card_combination'].count("card")):
        reshuffle_deck_if_empty(game)
        game.players[option.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
                
def carry_out_abbot_beg(option, game):
    max_gold_player_index = max(enumerate([player.gold for player in game.players]), key=lambda x: x[1])[0]
    game.players[max_gold_player_index].gold -= 1
    game.get_player_from_role_id(4).gold += 1
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("begged")
            
def carry_out_cardinal(option, game):
    # Special way of building
    # Regular building
    game.players[option.attributes['perpetrator']].buildings.add_card(game.players[option.attributes['perpetrator']].hand.get_a_card_like_it(option.attributes['built_card']))
    game.players[option.attributes['perpetrator']].gold -= option.attributes['built_card'].cost-option.attributes['factory']
    game.players[option.attributes['perpetrator']].gold = max(0, game.players[option.attributes['perpetrator']].gold)
    if option.attributes['replica']:
        game.players[option.attributes['perpetrator']].replicas = option.attributes['replica']

    # Cardinal card take
    if option.attributes['cards_to_give']:
        game.players[option.attributes['target']].gold -= len(option.attributes['cards_to_give'])
        for card in option.attributes['cards_to_give']:
            game.players[option.attributes['target']].hand.add_card(game.players[option.attributes['perpetrator']].hand.get_a_card_like_it(card))
            
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

    # ID 5
def carry_out_merchant(option, game):
    for building in game.players[option.attributes['perpetrator']].buildings.cards:
        if building.suit == "trade":
            game.players[option.attributes['perpetrator']].gold += 1
    game.players[option.attributes['perpetrator']].gold += 1
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
    
def carry_out_alchemist(option, game):
    pass
    # Cost is zero

def carry_out_trader(option, game):
    for building in game.players[option.attributes['perpetrator']].buildings.cards:
        if building.suit == "trade":
            game.players[option.attributes['perpetrator']].gold += 1
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

    # ID 6
def carry_out_architect(option, game):
    # Building Limit is three
    for _ in range(2):
        reshuffle_deck_if_empty(game)
        game.players[option.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
    
def carry_out_navigator(option, game):
    # Building Limit is zero
    if option.attributes['choice'] == "4card":
        for _ in range(4):
            reshuffle_deck_if_empty(game)
            game.players[option.attributes['perpetrator']].hand.add_card(game.deck.draw_card())
    if option.attributes['choice'] == "4gold":
        game.players[option.attributes['perpetrator']].gold += 4
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

def carry_out_scholar_draw(option, game):
    game.seven_drawn_cards = Deck(empty=True)
    for _ in range(min(7, len(game.deck.cards))):
        reshuffle_deck_if_empty(game)
        card = game.deck.draw_card()
        game.players[option.attributes['perpetrator']].hand.add_card(card)
        game.seven_drawn_cards.add_card(card)
        
    game.gamestate.state = 9
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
    game.gamestate.next_gamestate = GameState(state=5, player_id=option.attributes['perpetrator'], already_done_moves=game.gamestate.already_done_moves)

def carry_out_scholar_put_back(option, game):
    for card in option.attributes['unchosen_cards'].cards:
        game.deck.add_card(game.players[option.attributes['perpetrator']].hand.get_a_card_like_it(card))
    game.gamestate = game.gamestate.next_gamestate
    game.seven_drawn_cards = []

# ID 7
def carry_out_marshal(option, game):
    game.players[option.attributes['perpetrator']].gold -= option.attributes['choice'].cost
    game.players[option.attributes['target']].gold += option.attributes['choice'].cost
    game.players[option.attributes['perpetrator']].buildings.add_card(game.players[option.attributes['target']].buildings.get_a_card_like_it(option.attributes['choice']))
    if check_if_building_is_replica(game.players[option.attributes['target']], option.attributes['choice']):
        game.players[option.attributes['target']].replicas -= 1
    settle_museum(option, game)
    settle_lighthouse(option, game)
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

def carry_out_warlord(option, game):
    game.players[option.attributes['perpetrator']].gold -= option.attributes['choice'].cost-1
    game.discard_deck.add_card(game.players[option.attributes['target']].buildings.get_a_card_like_it(option.attributes['choice']))
    
    if check_if_building_is_replica(game.players[option.attributes['target']], option.attributes['choice']):
        game.players[option.attributes['target']].replicas -= 1
        
    settle_museum(option, game)
    settle_lighthouse(option, game)
        
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")
    graveyard_owner = get_graveyard_owner(game)
    if graveyard_owner is not None and graveyard_owner != game.players[option.attributes['perpetrator']]:
        game.gamestate.state = 6
        game.gamestate.player_id = graveyard_owner.id
        game.gamestate.interruption = True
        game.gamestate.next_gamestate = GameState(state=5, player_id=option.attributes['perpetrator'], already_done_moves=["character_ability"])


def carry_out_diplomat(option, game):
    game.players[option.attributes['perpetrator']].gold -= option.attributes['money_owed']
    game.players[option.attributes['target']].gold += option.attributes['money_owed']
    game.players[option.attributes['perpetrator']].buildings.add_card(game.players[option.attributes['target']].buildings.get_a_card_like_it(option.attributes['choice']))
    game.players[option.attributes['target']].buildings.add_card(game.players[option.attributes['perpetrator']].buildings.get_a_card_like_it(option.attributes['give']))
    
    if check_if_building_is_replica(game.players[option.attributes['target']], option.attributes['choice']):
        game.players[option.attributes['target']].replicas -= 1
    settle_museum(option, game)
    settle_lighthouse(option, game)
    
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("character_ability")

def carry_out_take_gold_for_war(option, game):
    for building in game.players[option.attributes['perpetrator']].buildings.cards:
        if building.suit == "war":
            game.players[option.attributes['perpetrator']].gold += 1
    game.gamestate.state = 5
    game.gamestate.player_id = option.attributes['perpetrator']
    game.gamestate.already_done_moves.append("take_gold")




def reshuffle_deck_if_empty(game):
    if not len(game.deck.cards):
        if not len(game.discard_deck.cards):
            return
        game.discard_deck.shuffle_deck()
        game.deck = deepcopy(game.discard_deck)
        game.discard_deck = Deck(empty=True)


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

def confirm_role_knowledges(revealed_player, game):
    """
    Reveal the role of the player_character to all players.
    """
    # If blackmailer reveales heroption first, I know witch isn't picked
    logically_left_out_role_ids = [role_id for role_id in game.used_roles if role_id < role_to_role_id[revealed_player.role]]
    for player in game.players:
        for role_knowledge in player.known_roles:
            if role_knowledge.player_id == revealed_player.id:
                # Confirm the role of the revealed player
                role_knowledge.possible_roles = {role_to_role_id[revealed_player.role]: revealed_player.role}
                role_knowledge.confirmed = True
            elif not role_knowledge.confirmed:
                # Remove the revealed role from the possible_roles of all other players and logically left out roles
                role_knowledge.possible_roles = {role_id: role for role_id, role in role_knowledge.possible_roles.items() if role != revealed_player.role or role_id not in logically_left_out_role_ids}


def move_crown(game, target_player_id):
    for player in game.players:
        if player.crown:
            player.crown = False
            break
    game.players[target_player_id].crown = True
    troneroom_owner_gold(game)
    
    
def get_action_map():
    
    action_map = {
    "role_pick": carry_out_role_pick,
    "gold_or_card": carry_out_gold_or_card,
    "which_card_to_keep": carry_out_put_back_card,
    "blackmail_response": carry_out_respond_to_blackmail,
    "reveal_blackmail_as_blackmailer": carry_out_responding_to_blackmail_response,
    "reveal_warrant_as_magistrate": carry_out_magistrate_reaveal,
    "build": carry_out_building,
    "empty_option": carry_out_empty,
    "finish_round": finish_main_sequnce_actions,
    "ghost_town_color_choice": carry_out_ghost_town,
    "smithy_choice": carry_out_smithy,
    "laboratory_choice": carry_out_laboratory,
    "magic_school_choice": carry_out_magic_school,
    "weapon_storage_choice": carry_out_weapon_storage,
    "lighthouse_choice": carry_out_lighthouse,
    "museum_choice": carry_out_museum,
    "graveyard": carry_out_graveyard,
    "take_gold_for_war": carry_out_take_gold_for_war,
    "assassination": carry_out_assasination,
    "magistrate_warrant": carry_out_warranting,
    "bewitching": carry_out_bewitching,
    "steal": carry_out_stealing,
    "blackmail": carry_out_blackmail,
    "spy": carry_out_spying,
    "magic_hand_change": carry_out_magicking,
    "discard_and_draw": carry_out_magicking,
    "look_at_hand": carry_out_wizard_hand_looking,
    "take_from_hand": carry_out_wizard_take_from_hand,
    "seer": carry_out_seer_take_a_card,
    "give_back_card": carry_out_seer_give_back_cards,
    "take_crown_king": carry_out_take_crown_king,
    "give_crown": carry_out_emperor,
    "take_crown_pat": carry_out_take_crown_patrician,
    "bishop": carry_out_bishop,
    "cardinal_exchange": carry_out_cardinal,
    "abbot_gold_or_card": carry_out_abbot,
    "abbot_beg": carry_out_abbot_beg,
    "merchant": carry_out_merchant,
    "alchemist": carry_out_alchemist,
    "trader": carry_out_trader,
    "architect": carry_out_architect,
    "navigator_gold_card": carry_out_navigator,
    "scholar": carry_out_scholar_draw,
    "scholar_card_pick": carry_out_scholar_put_back,
    "warlord_desctruction": carry_out_warlord,
    "marshal_steal": carry_out_marshal,
    "diplomat_exchange": carry_out_diplomat
    }
    
    return action_map