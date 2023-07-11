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
    def __init__(self, current_player=None, state = 0, already_done_moves=[], next_game_state= None) -> None:
        self.current_player = None
        self.state = 0
        # Moves such that can be described as "You can do it anytime x times."
        self.already_done_moves = []
        # For gamestate where just from the state is unclear whats next,
        # like reaction decisions, for example reaveal or not as blackmailer
        self.next_game_state = None

# Confidence: 5 Surely know everyhing, 4 already used a card, 3 used two cards, ... 0
# id player_id, -1 means the deck
class HandKnowlage():
    def __init__(self, player_id, hand, confidence) -> None:
        self.player_id = player_id
        self.confidence = confidence
        self.hand = hand