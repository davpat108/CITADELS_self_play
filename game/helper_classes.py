class RolePropery():
    def __init__(self) -> None:
        self.dead = False
        self.warrant = None
        self.possessed = False
        self.robbed = False
        self.blackmail = None
        
    def reset_role_properties(self):
        self.dead = False
        self.warrant = None
        self.possessed = False
        self.robbed = False
        self.blackmail = None

class GameState():
    def __init__(self, player=None, state = 0, already_done_moves=None, next_gamestate= None) -> None:
        self.player = player
        self.state = state
        # Moves such that can be described as "You can do it anytime x times."
        self.already_done_moves = already_done_moves if already_done_moves is not None else []
        # For gamestate where just from the state is unclear whats next,
        # like reaction decisions, for example reaveal or not as blackmailer
        self.next_gamestate = next_gamestate

# Confidence: 5 Surely know everyhing, 4 already used a card, 3 used two cards, ... 0
# id player_id, -1 means the deck
class HandKnowlage():
    def __init__(self, player_id, hand, confidence) -> None:
        self.player_id = player_id
        self.confidence = confidence
        self.hand = hand