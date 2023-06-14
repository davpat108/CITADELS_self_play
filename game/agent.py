from deck import Deck

class Agent():

    def __init__(self, id:int) -> None:
        # Game start
        self.hand = Deck(empty=True)
        self.role = None
        self.buildings = Deck(empty=True)
        self.warrant = False
        self.blackmail = False
        self.crown = False
        self.money = 2
        self.id = id