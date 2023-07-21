import numpy as np

class CFRNode:
    def __init__(self, game, num_actions, player_id, parent=None, player_count = 6):
        self.game = game
        self.num_actions = num_actions

        self.parent = parent
        self.children = [] # (Option that carries the game to the node, NODE)
        self.player_id = player_id

        
        # Initialize regrets and strategy
        self.cumulative_regrets = np.zeros(len(self.children))
        self.strategy = np.zeros(len(self.children))
        self.cumulative_strategy = np.zeros(len(self.children))
        self.node_value = np.zeros(player_count) # For the current_player node_value[player_id] = reward
        

    def action_choice(self):
        # returns a child node, so node and option
        pass

    def expand(self):
        # Account for player
        pass

    def is_terminal(self):
        if self.game.is_terminal():
            return True

    def cfr(self, max_iterations=1000):
        # If terminal end the game
        if self.is_terminal():
            return 

        if self.children == []:
            self.expand()
        
        node = self
        for iteration in range(max_iterations):
            # Traverse
            node, option = node.action_choice()

            # leaf node
            if node.children == []:
                node.expand()
                # terminal node, get rewards, calc regrets, backpropagate
                if node.is_terminal():
                    reward = node.get_reward()
                    node.backpropagate(reward) # backpropagate the reward and calculate regrets
                    node.update_strategy()
                    node = self
        

    def update_regrets(self):
        # backprops, checks all the other choices it could have made from the parent and calcs reward
        # pay attention to player id

        player_id = self.player_id
        # Calculate the actual rewards for each action
        actual_rewards = [child.node_value[player_id] for child in self.children]
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
        self.update_regrets(reward)

        # Recursively call backpropagate on parent node
        self.parent.backpropagate(reward)

    def update_strategy(self):
        total_regret = np.sum(self.cumulative_regrets)

        if total_regret > 0:
            # Normalize the regrets to get a probability distribution
            self.strategy = self.cumulative_regrets / total_regret
        else:
            # If there is no regret, use a uniform random strategy
            self.strategy = np.ones(self.num_actions) / self.num_actions

        # Update the cumulative strategy used for the average strategy output
        self.cumulative_strategy += self.strategy
    
