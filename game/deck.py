from copy import deepcopy
from random import shuffle

import numpy as np


class Card:
    def __init__(self, type_ID:int, suit: str,  cost:int):
        self.suit = suit
        self.type_ID = type_ID
        self.cost = cost

    def __eq__(self, other:int) -> bool:
        if isinstance(other, Card):
            return self.type_ID == other.type_ID
        return False

    def __lt__(self, other:int) -> bool:
        if isinstance(other, Card):
            return self.type_ID < other.type_ID
        return False
    
    def __deepcopy__(self, memo):
        return Card(deepcopy(self.type_ID, memo), deepcopy(self.suit, memo), deepcopy(self.cost, memo))
    
class Deck:
    def __init__(self, used_cards=None, empty = False):
        if not empty:
            self.cards = []
            for card_info in used_cards:
                self.cards.append(Card(**card_info))
        
        else:
            self.cards = []

        self.shuffle_deck()

    def __deepcopy__(self, memo):
        new_deck = Deck(empty=True)
        new_deck.cards = [deepcopy(card, memo) for card in self.cards]
        return new_deck
    
    def __eq__(self, other):
        if isinstance(other, Deck):
            return self.cards == other.cards
        return False

    
    def get_a_card_like_it(self, card_to_get:Card):
        for card in self.cards:
            if card.type_ID == card_to_get.type_ID:
                self.cards.remove(card)
                return card
        # Turn off the check, some cards go missing during sampling
        return card_to_get
        raise KeyError("No card like that in the deck")

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
            pass
            #logging.debug("Not a card, deck empty")
    
    def shuffle_deck(self):
        shuffle(self.cards)

    def encode_deck(self):
        encoded_type_IDs = np.zeros(40, dtype=int)
        encoded_suits = np.zeros(5, dtype=int)
        suit_to_index = {
            "trade": 0,
            "war": 1,
            "religion": 2,
            "lord": 3,
            "unique": 4
        }

        for card in self.cards:
            encoded_type_IDs[card.type_ID] += 1
            encoded_suits[suit_to_index[card.suit]] += 1

        return encoded_type_IDs, encoded_suits