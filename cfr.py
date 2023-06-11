import numpy as np

class CFRNode:
    def __init__(self, state, num_actions, player_id, parent=None, policy=None, cfr_strategy_func=None, action_choice_func=None):
        self.state = state
        self.num_actions = num_actions

        self.parent = parent
        self.children = [] # (Option that carries the game to the node, NODE)
        self.player_id = player_id

        
        # Initialize regrets and strategy
        self.cumulative_regrets = np.zeros(len(self.children))
        self.strategy = np.zeros(len(self.children))
        self.cumulative_strategy = np.zeros(len(self.children))
        
        # Optionally initialize a policy
        self.policy = np.ones(len(self.children)) / num_actions if policy is None else policy
        
        # Initialize strategy update and action choice functions
        self.cfr_strategy_func = self.default_cfr_strategy if cfr_strategy_func is None else cfr_strategy_func
        self.action_choice_func = self.default_action_choice if action_choice_func is None else action_choice_func

    def default_cfr_strategy(self, cumulative_regrets):
        # Implement the default strategy update logic here
        pass

    def action_choice(self):
        # returns a child node, so node and option
        pass

    def cfr(self, max_iterations=1000):
        # If terminal end the game
        if self.is_terminal():
            return 

        if self.children == []:
            self.expand()
        
        node = self
        for iteration in range(max_iterations):
            # Traverse
            node.update_strategy() 
            node, option = node.action_choice()

            # leaf node
            if node.children == []:
                node.expand()
                # terminal node, get rewards, calc regrets, backpropagate
                if node.children == []:
                    reward = node.get_reward()
                    node.backpropagate(reward) 
                    node.update_regrets() #All the way up to root
                    node = self
        
        return self.action_choice()


        
    def update_regrets(self):
        # Implement the logic to update regrets here
        pass
    
    def get_reward(self):
        # Implement the logic to get the reward here
        pass

    def backpropagate(self, reward):
        # Implement the logic to backpropagate the reward here
        pass

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
    
    def expand(self):
        # Account for player
        pass