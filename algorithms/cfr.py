import numpy as np
from copy import deepcopy
from game.checks import check_same_memory_address

class CFRNode:
    def __init__(self, game, current_player_id, original_player_id, parent=None, player_count = 6):
        self.game = game # Unkown informations are present but counted as dont care

        self.parent = parent
        self.children = [] # (Option that carries the game to the node, NODE)
        self.current_player_id = current_player_id
        # The player id of the player that started the game
        # It is used to know which player doesn't need private info sampling
        self.original_player_id = original_player_id

        
        # Initialize regrets and strategy
        self.cumulative_regrets = np.array([])
        self.strategy = np.zeros([])
        self.cumulative_strategy = np.zeros([])
        self.node_value = np.zeros(player_count) # For the current_player node_value[player_id] = reward
        

    def action_choice(self):
        # Normalize the strategy to ensure it sums to 1 (due to numerical issues)
        normalized_strategy = self.strategy / self.strategy.sum()

        # Choose an action according to the normalized strategy
        choice_index = np.random.choice(range(len(self.children)), p=normalized_strategy)

        # Return the chosen child and option
        return  self.children[choice_index][1], self.children[choice_index][0]


    def expand(self):
        if self.current_player_id == self.original_player_id and len(self.children) == 0:
            options = self.game.get_options_from_state()
            for option in options:

                hypothetical_game = deepcopy(self.game)

                # Sample if not the same players turn as before
                if self.parent is None or  hypothetical_game.gamestate.player_id != self.parent.game.gamestate.player_id:
                    hypothetical_game.sample_private_information(hypothetical_game.players[self.original_player_id])
                option.carry_out(hypothetical_game)
                hypothetical_game.sample_private_info_after_role_pick_end(hypothetical_game.players[self.original_player_id])

                print("Added child info set from orig child player's role: ", hypothetical_game.players[hypothetical_game.gamestate.player_id].role, " ID: ", hypothetical_game.gamestate.player_id, "Action leading there: ", option.name )
                self.children.append((option, CFRNode(game=hypothetical_game, current_player_id=hypothetical_game.gamestate.player_id, original_player_id=self.original_player_id, parent=self)))


            self.cumulative_regrets = np.zeros(len(self.children))
            self.strategy = np.zeros(len(self.children))
            self.cumulative_strategy = np.zeros(len(self.children))

        elif self.current_player_id != self.original_player_id and len(self.children) < 10:
            hypothetical_game = deepcopy(self.game)
            
            # Sample if not the same players turn as before
            if self.parent is None or hypothetical_game.gamestate.player_id != self.parent.game.gamestate.player_id:
                hypothetical_game.sample_private_information(hypothetical_game.players[self.original_player_id])
            options = hypothetical_game.get_options_from_state()
            choice_index = np.random.choice(range(len(options)))
            options[choice_index].carry_out(hypothetical_game)
            
            hypothetical_game.sample_private_info_after_role_pick_end(hypothetical_game.players[self.original_player_id])
            print("Added child info set from opponent child player's role: ", hypothetical_game.players[hypothetical_game.gamestate.player_id].role, " ID: ", hypothetical_game.gamestate.player_id, "Action leading there: ", options[choice_index].name )
            self.children.append((options[choice_index], CFRNode(game=hypothetical_game, current_player_id=hypothetical_game.gamestate.player_id, original_player_id=self.original_player_id, parent=self)))

            self.cumulative_regrets = np.append(self.cumulative_regrets, 0)
            self.strategy = np.append(self.strategy, 0)
            self.cumulative_strategy = np.append(self.cumulative_strategy, 0)

            

    def is_terminal(self):
        return self.game.terminal

    def get_reward(self):
        return self.game.rewards

    def cfr(self, max_iterations=100):
        # If the cfr is called from a terminal node, return
        if self.is_terminal():
            return 

        self.expand()
        
        node = self
        for i in range(max_iterations):
            # Traverse
            node.update_strategy()
            node, action = node.action_choice()
            print(f"cfr{i}, Traversion: ", action.name)
            # leaf node
            # terminal node, get rewards, calc regrets, backpropagate
            if node.is_terminal():
                reward = node.get_reward()
                node.backpropagate(reward) # backpropagate the reward and calculate regrets
                node.update_strategy()
                node = self
            else:
                node.expand()
        

        

    def update_regrets(self):
        # backprops, checks all the other choices it could have made from the parent and calcs reward
        # pay attention to player id

        player_id = self.current_player_id
        # Calculate the actual rewards for each action
        actual_rewards = [child[1].node_value[player_id] for child in self.children]
        # Get the maximum possible reward
        max_reward = max(actual_rewards)
        # Update regrets
        for a in range(len(self.children)):
            self.cumulative_regrets[a] += max_reward - actual_rewards[a]
    

    def backpropagate(self, reward):
        if self.parent is None:
            return

        # Update the value of this node
        self.node_value += reward

        # Calculate regret for this node
        if self.children:
            self.update_regrets()

        # Recursively call backpropagate on parent node
        self.parent.backpropagate(reward)

    def update_strategy(self):
        total_regret = np.sum(self.cumulative_regrets)

        if total_regret > 0:
            # Normalize the regrets to get a probability distribution
            self.strategy = self.cumulative_regrets / total_regret
        else:
            # If there is no regret, use a uniform random strategy
            self.strategy = np.ones(len(self.children)) / len(self.children)

        # Update the cumulative strategy used for the average strategy output
        self.cumulative_strategy += self.strategy
    
