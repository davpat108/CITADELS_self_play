from random import shuffle, sample
from game.config import building_cards, unique_building_cards
import logging
class Card:
    def __init__(self, type_ID:int, suit: str,  cost:int):
        self.suit = suit
        self.type_ID = type_ID
        self.cost = cost

    def __eq__(self, other:int) -> bool:
        if isinstance(other, Card):
            return self.type_ID == other.type_ID
        return False

class Deck:
    def __init__(self, used_cards=None, empty = False):
        if not empty:
            if not used_cards:
                used_cards = building_cards + sample(unique_building_cards, 14)
                self.cards = []
                for card_info in used_cards:
                    self.cards.append(Card(**card_info))
            else:
                self.cards = []
                for card_info in used_cards:
                    self.cards.append(Card(**card_info))
        
        else:
            self.cards = []

        self.shuffle_deck()

    
    def get_a_card_like_it(self, card_to_get:Card):
        for card in self.cards:
            if card.type_ID == card_to_get.type_ID:
                self.cards.remove(card)
                return card
        raise Exception("No card like that in the deck")

    def draw_card(self):
        if len(self.cards) == 0:
            return "Deck Empty"
        return self.cards.pop(0)
    
    def add_card(self, card: Card):
        """
        Add a card to the end, mainly used for discard deck
        """
        if isinstance(card, Card):
            self.cards.append(card)
        else:
            logging.debug("Not a card, deck empty")
    
    def shuffle_deck(self):
        shuffle(self.cards)