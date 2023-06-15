from deck import Deck
from option import option

class Agent():

    def __init__(self, id:int) -> None:
        # Game start
        self.hand = Deck(empty=True)
        self.role = None
        self.buildings = Deck(empty=True)
        self.warrant_fake = False
        self.warrant_true = False
        self.blackmail_fake = False
        self.blackmail_true = False
        self.crown = False
        self.money = 2
        self.id = id

    def choose_role_options(self, avaible_roles:list) -> list:
        options = [option(choice=role, name="Role") for role in avaible_roles]
        return options
    
    def gold_or_card(self) -> list:
        options = [option(choice="gold", name="gold_or_card"), option(choice="gold", name="gold_or_card")]
        return options
    
    def blackmail_response_options(self, ) -> list:
        if self.blackmail_fake or self.blackmail_true:
            return [option(choice="pay", name="blackmail_response"), option(choice="not_pay", name="blackmail_response")]
        return None