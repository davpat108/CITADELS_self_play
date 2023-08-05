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
    def __init__(self, player_id=None, state = 0, already_done_moves=None, next_gamestate= None, interruption=False) -> None:
        self.player_id = player_id
        self.state = state
        # Moves such that can be described as "You can do it anytime x times."
        self.already_done_moves = already_done_moves if already_done_moves is not None else []
        # For gamestate where just from the state is unclear whats next,
        # like reaction decisions, for example reaveal or not as blackmailer
        self.next_gamestate = next_gamestate

        # Next is an interrupting decision if true, like blackmail reveal, magistrate, or graveyard
        self.interruption = interruption

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, GameState):
            return self.state == __value.state and self.player_id.id == __value.player_id.id
        return False

# Confidence: 5 Surely know everyhing, 4 already used a card, 3 used two cards, ... 0
# id player_id, -1 means the deck
# used attribute is only used when sampling information
class HandKnowledge():
    def __init__(self, player_id, hand, confidence, wizard=False) -> None:
        self.player_id = player_id
        self.confidence = confidence
        self.hand = hand
        self.wizard = wizard
        self.used = False

class RoleKnowlage():
    
    def __init__(self, player_id, possible_roles) -> None:
        self._player_id = player_id
        self._possible_roles = possible_roles
    
    @property
    def player_id(self):
        return self._player_id

    @player_id.setter
    def player_id(self, value):
        #print(f'Changing value of player_Id from {self._player_id} to {value}')
        self._player_id = value

    @property
    def possible_roles(self):
        return self._possible_roles

    @possible_roles.setter
    def possible_roles(self, value):
        #print(f'Changing value of possible roles from {self._possible_roles} to {value}')
        self._possible_roles = value