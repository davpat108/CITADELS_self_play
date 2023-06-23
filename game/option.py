class option():
    def __init__(self, name, **kwargs):
        self.name = name
        self.attributes = kwargs


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
        game.role_properties[self.attributes['target']].robbed = True
    
    def carry_out_blackmail(self, game):
        game.role_properties[self.attributes['real_target']].blackmail = "Real"
        game.role_properties[self.attributes['fake_target']].blackmail = "Fake"
        
    def carry_out_spying(self, game):
        cards_to_draw = sum([1 if building.suit == self.attributes['suit'] else 0 for building in self.attributes['target'].hand.cards])
        gold_to_steal = min(cards_to_draw, self.attributes['target'].gold)

        self.attributes['perpetrator'].gold += gold_to_steal
        self.attributes['target'].gold -= gold_to_steal

        self.attributes['perpetrator'].hand.add_card(game.deck.draw_card())

    # ID 2


    

        
        
        


def get_player_from_role(self, role_id, game):
    for player in game.players:
        if player.role == game.roles[role_id]:
            return player
    return None