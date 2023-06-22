class option():
    def __init__(self, name, **kwargs):
        self.name = name
        self.attributes = kwargs


    def carry_out_action(self, game):
        pass

    def carry_out_assasination(self, game):
        pass


def get_player_from_role(self, role_id, game):
    for player in game.players:
        if player.role == game.roles[role_id]:
            return player
    return None