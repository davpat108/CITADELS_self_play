from copy import deepcopy
from game.deck import Deck

class option():
    def __init__(self, name, **kwargs):
        self.name = name
        self.attributes = kwargs

    def __eq__(self, other):
        return self.name == other.name and self.attributes == other.attributes

    def carry_out_action(self, game):
        pass

    # ID 0
    def carry_out_assasination(self, game):
        game.role_properties[self.attributes['target']].dead = True
        
    def carry_out_warranting(self, game):
        game.role_properties[self.attributes['real_target']].warrant = "real"
        game.role_properties[self.attributes['fake_targets'][0]].warrant = "fake"
        game.role_properties[self.attributes['fake_targets'][1]].warrant = "fake"
        
    def carry_out_bewitching(self, game):
        game.role_properties[self.attributes['target']].possessed = True
        
    # ID 1
    def carry_out_stealing(self, game):
        if not game.role_properties[1].dead and not game.role_properties[1].possessed:
            game.role_properties[self.attributes['target']].robbed = True
    
    def carry_out_blackmail(self, game):
        if not game.role_properties[1].dead and not game.role_properties[1].possessed:
            game.role_properties[self.attributes['real_target']].blackmail = "Real"
            game.role_properties[self.attributes['fake_target']].blackmail = "Fake"
        
    def carry_out_spying(self, game):
        if not game.role_properties[1].dead and not game.role_properties[1].possessed:
            cards_to_draw = sum([1 if building.suit == self.attributes['suit'] else 0 for building in self.attributes['target'].hand.cards])
            gold_to_steal = min(cards_to_draw, self.attributes['target'].gold)

            self.attributes['perpetrator'].gold += gold_to_steal
            self.attributes['target'].gold -= gold_to_steal


            reshuffle_deck_if_empty(game)
            self.attributes['perpetrator'].hand.add_card(game.deck.draw_card())

    # ID 2
    def carry_out_magicking(self, game):
        if not game.role_properties[2].dead and not game.role_properties[2].possessed:
            if self.name == "magic_hand_change":
                self.attributes['perpetrator'].hand.cards, self.attributes['target'].hand.cards = self.attributes['target'].hand.cards, self.attributes['perpetrator'].hand.cards

            if self.name == "discard_and_draw":
                for card in self.attributes['perpetrator'].cards:
                    game.deck.add_card(self.attributes['perpetrator'].hand.get_a_card_like_it(card))
                for _ in range(len(self.attributes['perpetrator'].cards)):
                    reshuffle_deck_if_empty(game)
                    self.attributes['perpetrator'].hand.add_card(game.deck.draw_card())

    def carry_out_wizard_hand_looking(self, game):
        pass

    def carry_out_wizard_take_from_hand(self, game):
        if not game.role_properties[2].dead and not game.role_properties[2].possessed:
            # Can be the same building you already have, unlike with regualr building
            if self.attributes['build']:
                self.attributes['perpetrator'].buildings.add_card(self.attributes['target'].hand.get_a_card_like_it(self.attributes['card']))
                self.attributes['perpetrator'].gold -= self.attributes['card'].cost
            else:
                self.attributes['perpetrator'].hand.add_card(self.attributes['target'].hand.get_a_card_like_it(self.attributes['card']))

    def carry_out_seer_take_a_card(self, game):
        if not game.role_properties[2].dead and not game.role_properties[2].possessed:
            for player in game.players:
                if player.id != self.attributes['perpetrator'].id:
                    player.hand.shuffle_deck()
                    reshuffle_deck_if_empty(game)
                    self.hand.add_card(player.hand.draw_card())

    def carry_out_give_back_cards(self, game):
        if not game.role_properties[2].dead and not game.role_properties[2].possessed:
            for handout in self.attributes['card_handouts'].items():
                handout[0].hand.add_card(self.attributes['perpetrator'].hand.get_a_card_like_it(handout[1]))

    # ID 3
    def carry_out_take_crown_king(self, game):
        if not game.role_properties[3].dead and not game.role_properties[3].possessed:
            for player in game.players:
                if player.crown:
                    player.crown = False
                    break
            for building in self.attributes['perpetrator'].buildings.cards:
                if building.suit == "lord":
                    self.attributes['perpetrator'].gold += 1

        self.attributes['perpetrator'].crown = True

    def carry_out_take_crown_patrician(self, game):
        if not game.role_properties[3].dead and not game.role_properties[3].possessed:
            for player in game.players:
                if player.crown:
                    player.crown = False
                    break
            for building in self.attributes['perpetrator'].buildings.cards:
                if building.suit == "lord":
                    reshuffle_deck_if_empty(game)
                    self.attributes['perpetrator'].hand.add_card(game.deck.draw_card())
                
        self.attributes['perpetrator'].crown = True

    def carry_out_emperor(self, game):
        if not game.role_properties[3].dead and not game.role_properties[3].possessed:
            for building in self.attributes['perpetrator'].buildings.cards:
                if building.suit == "lord":
                    self.attributes['perpetrator'].gold += 1
            
            if self.attributes['gold_or_card'] == "card":
                self.attributes['target'].hand.shuffle_deck()
                self.attributes['perpetrator'].hand.add_card(self.attributes['target'].hand.draw_card())

            if self.attributes['gold_or_card'] == "gold":
                self.attributes['perpetrator'].gold += 1
                self.attributes['target'].gold -= 1

        if not game.role_properties[3].possessed:
            self.attributes['perpetrator'].crown = False
            self.attributes['target'].crown = True

    # ID 4
    def carry_out_bishop(self, game):
        if not game.role_properties[4].dead and not game.role_properties[4].possessed:
            for building in self.attributes['perpetrator'].buildings.cards:
                if building.suit == "religion":
                    self.attributes['perpetrator'].gold += 1

    
    def carry_out_abbot(self, game):
        if not game.role_properties[4].dead and not game.role_properties[4].possessed:
            self.attributes['perpetrator'].gold += self.attributes['gold_or_card_combination'].count("gold")
            for _ in range(self.attributes['gold_or_card_combination'].count("card")):
                reshuffle_deck_if_empty(game)
                self.attributes['perpetrator'].hand.add_card(game.deck.draw_card())

    def carry_out_cardinal(self, game):
        # Special way of building

        if not game.role_properties[4].dead and not game.role_properties[4].possessed:
            self.attributes['perpetrator'].buildings.add_card(self.attributes['perpetrator'].hand.get_a_card_like_it(self.attributes['built_card']))
            self.attributes['perpetrator'].gold -= self.attributes['built_card'].cost
            self.attributes['perpetrator'].gold = max(0, self.attributes['perpetrator'].gold)

            if self.attributes['cards_to_give']:
                self.attributes['target'].gold -= len(self.attributes['cards_to_give'])
                for card in self.attributes['cards_to_give']:
                    self.attributes['target'].hand.add_card(self.attributes['perpetrator'].hand.get_a_card_like_it(card))
    
    # ID 5
    def carry_out_merchant(self, game):
        if not game.role_properties[5].dead and not game.role_properties[5].possessed:
            for building in self.attributes['perpetrator'].buildings.cards:
                if building.suit == "trade":
                    self.attributes['perpetrator'].gold += 1
            self.attributes['perpetrator'].gold += 1
    
    def carry_out_alchemist(self, game):
        pass
        # TODO: Implement alchemist after regular build implemented

    def carry_out_trader(self, game):
        if not game.role_properties[5].dead and not game.role_properties[5].possessed:
            for building in self.attributes['perpetrator'].buildings.cards:
                if building.suit == "religion":
                    self.attributes['perpetrator'].gold += 1

    # ID 6
    def carry_out_architect(self, game):
        # Building Limit is three
        if not game.role_properties[6].dead and not game.role_properties[6].possessed:
            for _ in range(2):
                reshuffle_deck_if_empty(game)
                self.attributes['perpetrator'].hand.add_card(game.deck.draw_card())

    def carry_out_navigator(self, game):
        # Building Limit is zero
        if not game.role_properties[6].dead and not game.role_properties[6].possessed:
            if self.attributes['choice'] == "4card":
                for _ in range(4):
                    reshuffle_deck_if_empty(game)
                    self.attributes['perpetrator'].hand.add_card(game.deck.draw_card())
            if self.attributes['choice'] == "4gold":
                self.attributes['perpetrator'].gold += 4

    def carry_out_scholar_draw(self, game):
        if not game.role_properties[6].dead and not game.role_properties[6].possessed:
            for _ in range(7):
                reshuffle_deck_if_empty(game)
                self.attributes['perpetrator'].hand.add_card(game.deck.draw_card())

    def carry_out_scholar_put_back(self, game):
        if not game.role_properties[6].dead and not game.role_properties[6].possessed:
            for card in self.attributes['unchosen_cards']:
                game.deck.add_card(self.attributes['perpetrator'].hand.get_a_card_like_it(card))

    
    def carry_out_marshal(self, game):
        if not game.role_properties[7].dead and not game.role_properties[7].possessed:
            self.attributes['perpetrator'].gold -= self.attributes['choice'].cost
            self.attributes['target'].gold += self.attributes['choice'].cost
            self.attributes['perpetrator'].buildings.add_card(self.attributes['target'].buildings.get_a_card_like_it(self.attributes['choice']))

    def carry_out_warlord(self, game):
        if not game.role_properties[7].dead and not game.role_properties[7].possessed:
            self.attributes['perpetrator'].gold -= self.attributes['choice'].cost-1
            self.attributes['target'].gold += self.attributes['choice'].cost-1
            game.discard_deck.add_card(self.attributes['target'].buildings.get_a_card_like_it(self.attributes['choice']))

    def carry_out_diplomat(self, game):
        if not game.role_properties[7].dead and not game.role_properties[7].possessed:
            self.attributes['perpetrator'].gold -= self.attributes['money_owed'].cost
            self.attributes['target'].gold += self.attributes['money_owed'].cost
            self.attributes['perpetrator'].buildings.add_card(self.attributes['target'].buildings.get_a_card_like_it(self.attributes['take']))
            self.attributes['target'].buildings.add_card(self.attributes['perpetrator'].buildings.get_a_card_like_it(self.attributes['give']))


def reshuffle_deck_if_empty(game):
    if not len(game.deck.cards):
        game.discard_deck.shuffle_deck()
        game.deck = deepcopy(game.discard_deck)
        game.discard_deck = Deck(empty=True)

def get_player_from_role(self, role_id, game):
    for player in game.players:
        if player.role == game.roles[role_id]:
            return player
    return None