import numpy as np
from copy import deepcopy
from game.checks import check_same_memory_address

class CFRNode:
    def __init__(self, game, current_player_id, original_player_id, parent=None, player_count = 6, role_pick_node=False):
        self.game = game # Unkown informations are present but counted as dont care

        self.parent = parent
        self.children = [] # (Option that carries the game to the node, NODE)
        self.current_player_id = current_player_id
        # The player id of the player that started the game
        # It is used to know which player doesn't need private info sampling
        self.original_player_id = original_player_id

        
        # Initialize regrets and strategy
        self.cumulative_regrets = np.array([])
        self.strategy = np.array([])
        self.cumulative_strategy = np.array([])
        self.node_value = np.zeros(player_count)

        self.role_pick_node = role_pick_node
        

    def weighted_average_strategy(self, strategy_matrix, pick_hierarchy):
        """
        strategy_matrix: A numpy array of shape (player_count, len(children)).
        pick_hierarchy: A list of player indexes representing the order of picking.

        Returns a 1-D numpy array of the weighted average strategy.
        """
        player_count = len(pick_hierarchy)
        weighted_strategy = np.zeros(strategy_matrix.shape[1])

        for i, player_idx in enumerate(pick_hierarchy):
            weight = player_count - i
            weighted_strategy += strategy_matrix[player_idx] * weight

        return weighted_strategy / sum(pick_hierarchy)

    def action_choice(self):
        if not self.role_pick_node:
            # Normalize the strategy to ensure it sums to 1 (due to numerical issues)
            normalized_strategy = self.strategy / self.strategy.sum()

            # Choose an action according to the normalized strategy
            choice_index = np.random.choice(range(len(self.children)), p=normalized_strategy)

            # Return the chosen child and option
            return self.children[choice_index][1], self.children[choice_index][0]

        else:
            pick_hierarchy = self.game.turn_orders_for_roles
            avg_strategy = self.weighted_average_strategy(self.strategy, pick_hierarchy)
    
            # Normalize the strategy to ensure it sums to 1 (due to numerical issues)
            normalized_strategy = avg_strategy / avg_strategy.sum()

            # Choose an action according to the normalized strategy
            choice_index = np.random.choice(range(len(self.children)), p=normalized_strategy)

            # Return the chosen child and option
            return self.children[choice_index][1], self.children[choice_index][0]

    def expand(self):
        
        if self.game.gamestate.state == 0 and not self.children:
            self.role_pick_node = True
            self.expand_role_pick()
        elif self.current_player_id == self.original_player_id and not self.children:
            self.expand_for_original_player()
        elif self.current_player_id != self.original_player_id and len(self.children) < 10:
            self.expand_for_opponents()


    def expand_role_pick(self):
        max_repeat_count = 10
        for i in range(max_repeat_count):
            hypothetical_game = deepcopy(self.game)
            while hypothetical_game.gamestate.state != 1:
                options = hypothetical_game.get_options_from_state()
                choice_index = np.random.choice(range(len(options)))
                if hypothetical_game.gamestate.player_id == self.original_player_id:
                    option_to_carry_out = options[choice_index]
                options[choice_index].carry_out(hypothetical_game)
            # Must be at the end of rolepick
            assert hypothetical_game.gamestate.player_id == hypothetical_game.get_player_from_role_id(hypothetical_game.used_roles[0]).id
            self.children.append((option_to_carry_out, CFRNode(game=hypothetical_game, current_player_id=hypothetical_game.gamestate.player_id, original_player_id=self.original_player_id, parent=self, role_pick_node=False)))
        
            self.cumulative_regrets = np.zeros((1, 6)) if self.cumulative_regrets.size == 0 else np.concatenate((self.cumulative_regrets, np.zeros((1, 6))), axis=0)
            self.strategy = np.zeros((1, 6)) if self.strategy.size == 0 else np.concatenate((self.strategy, np.zeros((1, 6))), axis=0)
            self.cumulative_strategy = np.zeros((1, 6)) if self.cumulative_strategy.size == 0 else np.concatenate((self.cumulative_strategy, np.zeros((1, 6))), axis=0)
        
        self.cumulative_regrets = self.cumulative_regrets.T
        self.strategy = self.strategy.T
        self.cumulative_strategy = self.cumulative_strategy.T
            
    
    def expand_for_original_player(self):
        options = self.game.get_options_from_state()
        for option in options:

            hypothetical_game = deepcopy(self.game)

            # Sample if not the same players turn as before
            if self.parent is None or hypothetical_game.gamestate.player_id != self.parent.game.gamestate.player_id:
                hypothetical_game.sample_private_information(hypothetical_game.players[self.original_player_id], role_sample=self.parent.game.gamestate.state != 0 if self.parent else False)
            option.carry_out(hypothetical_game)
            
            print("Added child info set from orig child player's role: ", hypothetical_game.players[hypothetical_game.gamestate.player_id].role, " ID: ", hypothetical_game.gamestate.player_id, "Action leading there: ", option.name)
            self.children.append((option, CFRNode(game=hypothetical_game, current_player_id=hypothetical_game.gamestate.player_id, original_player_id=self.original_player_id, parent=self, role_pick_node=hypothetical_game.gamestate.state == 0)))

        self.cumulative_regrets = np.zeros(len(self.children))
        self.strategy = np.zeros(len(self.children))
        self.cumulative_strategy = np.zeros(len(self.children))


    def expand_for_opponents(self):
        hypothetical_game = deepcopy(self.game)
        # Sample if not the same players turn as before
        if self.parent is None or hypothetical_game.gamestate.player_id != self.parent.game.gamestate.player_id:
            hypothetical_game.sample_private_information(hypothetical_game.players[self.original_player_id], role_sample=self.parent.game.gamestate.state != 0 if self.parent else False)

        options = hypothetical_game.get_options_from_state()
        choice_index = np.random.choice(range(len(options)))
        options[choice_index].carry_out(hypothetical_game)
        print("Added child info set from opponent child player's role: ", hypothetical_game.players[hypothetical_game.gamestate.player_id].role, " ID: ", hypothetical_game.gamestate.player_id, "Action leading there: ", options[choice_index].name)

        child_options = [child[0] for child in self.children]
        if not options[choice_index] in child_options:
            self.children.append((options[choice_index], CFRNode(game=hypothetical_game, current_player_id=hypothetical_game.gamestate.player_id, original_player_id=self.original_player_id, parent=self, role_pick_node=hypothetical_game.gamestate.state == 0)))

            self.cumulative_regrets = np.append(self.cumulative_regrets, 0)
            self.strategy = np.append(self.strategy, 0)
            self.cumulative_strategy = np.append(self.cumulative_strategy, 0)


    def is_terminal(self):
        return self.game.terminal

    def get_reward(self):
        return self.game.rewards

    def cfr(self, max_iterations=1000):
        # If the cfr is called from a terminal node, return
        if self.is_terminal():
            return 

        self.expand()
        
        node = self
        for i in range(max_iterations):
            # Traverse
            node.update_strategy()
            node, action = node.action_choice()
            if "role" in action.attributes.keys():
                print(f"cfr{i}, Traversion: ", action.name, "choice: ", action.attributes["role"])
            else:
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
        self.update_strategy()

        

    def update_regrets(self):
        # backprops, checks all the other choices it could have made from the parent and calcs reward
        # pay attention to player id
        if not self.role_pick_node:
            player_id = self.current_player_id
            # Calculate the actual rewards for each action
            actual_rewards = [child[1].node_value[player_id] for child in self.children]
            # Get the maximum possible reward
            max_reward = max(actual_rewards)
            # Update regrets
            for a in range(len(self.children)):
                self.cumulative_regrets[a] += max_reward - actual_rewards[a]
        else:
            actual_rewards = np.array([child[1].node_value for child in self.children]).T

            max_rewards = np.max(actual_rewards, axis=0)
            regret_values = max_rewards - actual_rewards
            self.cumulative_regrets += regret_values
    
    def print_tree_values(self, node, depth=0):
        """
        Recursively print the values of nodes in the tree.

        :param node: The current node being examined.
        :param depth: The depth of the current node in the tree. Root has depth 0.
        """

        # Print the node value with an indentation proportional to its depth
        print(f"Node depth: {depth} (Player {node.current_player_id}): Value = {node.node_value}")
        if depth > 20:
            return
        # Recursively call the function for each child
        for _, child_node in node.children:
            self.print_tree_values(child_node, depth + 1)

    def backpropagate(self, reward):
        # Update the value of this node
        self.node_value += reward

        # Calculate regret for this node
        if self.children:
            self.update_regrets()
            
        if self.parent is None:
            return
        # Recursively call backpropagate on parent node
        self.parent.backpropagate(reward)

    def update_strategy(self):
        if not self.role_pick_node:
            total_regret = np.sum(self.cumulative_regrets)
            if total_regret > 0:
                # Normalize the regrets to get a probability distribution
                self.strategy = self.cumulative_regrets / total_regret
            else:
                # If there is no regret, use a uniform random strategy
                self.strategy = np.ones(len(self.children)) / len(self.children)
        else:
                
            # Calculate the total regrets for each child
            total_regrets = np.sum(self.cumulative_regrets, axis=0)

            positive_regret_mask = total_regrets > 0
            uniform_strategy = np.ones(6) / 6
            new_strategies = np.tile(uniform_strategy[:, np.newaxis], (1, len(self.children)))
            new_strategies[:, positive_regret_mask] = self.cumulative_regrets[:, positive_regret_mask] / total_regrets[positive_regret_mask]
            self.strategy = new_strategies

        # Update the cumulative strategy used for the average strategy output
        self.cumulative_strategy += self.strategy
    