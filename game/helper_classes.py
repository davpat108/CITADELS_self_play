class RolePropery():
    def __init__(self) -> None:
        self.dead = False
        self.warrant = None
        self.possessed = False
        self.robbed = False
        self.blackmail = None
        
    def reset_at_turn_end(self):
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
