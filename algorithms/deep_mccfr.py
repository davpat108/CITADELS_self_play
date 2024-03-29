import numpy as np
from copy import deepcopy
import torch
from algorithms.train_utils import square_and_normalize, RanOutOfMemory
from random import randint


class CFRNode:
    def __init__(self, game, original_player_id, parent=None, player_count = 6, model=None, training=False, device="cuda:0", model_reward_weights=5, depth=0):
        self.device = device
        self.model = model
        if self.model:
            self.model.to(self.device)
        self.model_reward_weights = model_reward_weights
        self.original_player_id = original_player_id
        self.training = training

        self.depth = depth
        self.game = game # Unkown informations are present but counted as dont care
        self.skip_false_choice()
        
        self.parent = parent
        self.children = [] # (Option that carries the game to the node, NODE)
        self.current_player_id = game.gamestate.player_id

        # Initialize regrets and strategy
        self.cumulative_regrets = np.array([])
        self.strategy = np.array([])
        self.cumulative_strategy = np.array([])
        self.node_value = np.zeros(player_count)
        self.winning_probabilities = np.zeros(player_count)
        self.role_pick_node = game.gamestate.state == 0
        self.regret_gradient = float("inf")
        
        
        
    def skip_false_choice(self):
        """
        Checks if it would only have one children and if yes it would move the game on
        """
        i=0
        options = self.game.get_options_from_state()
        terminal = False
        while len(options) == 1 and not terminal:
            i+=1
            terminal = options[0].carry_out(self.game)
            options = self.game.get_options_from_state()
            if i > 100:
                terminal = True
            
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

    def action_choice(self, live=False):
        if not self.role_pick_node:
            # Normalize the strategy to ensure it sums to 1 (due to numerical issues)
            normalized_strategy = self.cumulative_strategy / self.cumulative_strategy.sum()
            choice_index = np.random.choice(range(len(self.children)), p=normalized_strategy)
            return self.children[choice_index][1], self.children[choice_index][0]

        elif live:
            return None, self.game.get_option_from_role_preference(self.strategy[self.game.gamestate.player_id])
        
        else:
            pick_hierarchy = self.game.turn_orders_for_roles
            avg_strategy = self.weighted_average_strategy(self.cumulative_strategy, pick_hierarchy)

            # Normalize the strategy to ensure it sums to 1 (due to numerical issues)
            if avg_strategy.sum() == 0:
                normalized_strategy = np.ones(len(self.children)) / len(self.children)
            else:
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
        for _ in range(max_repeat_count):
            hypothetical_game = deepcopy(self.game)
            while hypothetical_game.gamestate.state != 1:
                options = hypothetical_game.get_options_from_state()
                distribution = np.ones(len(options)) / len(options)
                choice_index = np.random.choice(range(len(options)), p=distribution)
                option_to_carry_out = options[choice_index]
                options[choice_index].carry_out(hypothetical_game)
        
            self.children.append((option_to_carry_out, CFRNode(game=hypothetical_game, original_player_id=self.original_player_id, parent=self, model=self.model, training=self.training, device=self.device, depth=self.depth+1)))

            self.cumulative_regrets = np.zeros((1, 6)) if self.cumulative_regrets.size == 0 else np.concatenate((self.cumulative_regrets, np.zeros((1, 6))), axis=0)
            self.strategy = np.zeros((1, 6)) if self.strategy.size == 0 else np.concatenate((self.strategy, np.zeros((1, 6))), axis=0)
            self.cumulative_strategy = np.zeros((1, 6)) if self.cumulative_strategy.size == 0 else np.concatenate((self.cumulative_strategy, np.zeros((1, 6))), axis=0)

        if self.model:
            for i in range(6):
                hypothetical_game = deepcopy(self.game)
                hypothetical_game.gamestate.player_id = i
                winning_probabilities = self.model_inference(hypothetical_game)

            if not self.training:
                self.pred_node_value = self.model_reward_weights * winning_probabilities
            

        self.cumulative_regrets = self.cumulative_regrets.T
        self.strategy = self.strategy.T
        self.cumulative_strategy = self.cumulative_strategy.T

    def expand_for_original_player(self):
        options = self.game.get_options_from_state()
        for option in options:

            hypothetical_game = deepcopy(self.game)

            if self.parent is None or hypothetical_game.gamestate.player_id != self.parent.game.gamestate.player_id:
                hypothetical_game.sample_private_information(hypothetical_game.players[self.original_player_id], role_sample=self.parent.game.gamestate.state != 0 if self.parent else False)
            option.carry_out(hypothetical_game)
            self.children.append((option, CFRNode(game=hypothetical_game, original_player_id=self.original_player_id, parent=self, model=self.model, training=self.training, device=self.device, depth=self.depth+1)))

        if self.model:
            winning_probabilities = self.model_inference(self.game)
            if not self.training:
                self.pred_node_value = self.model_reward_weights * winning_probabilities

        self.cumulative_regrets = np.zeros(len(self.children))
        self.strategy = np.zeros(len(self.children))
        self.cumulative_strategy = np.zeros(len(self.children))

    def expand_for_opponents(self):

        hypothetical_game = deepcopy(self.game)
        # Sample if not the same players turn as before
        if self.parent is None or hypothetical_game.gamestate.player_id != self.parent.game.gamestate.player_id:
            hypothetical_game.sample_private_information(hypothetical_game.players[self.original_player_id], role_sample=self.parent.game.gamestate.state != 0 if self.parent else False)

        options = hypothetical_game.get_options_from_state()
        if self.model:
            winning_probabilities = self.model_inference(hypothetical_game)

        distribution = np.ones(len(options)) / len(options)

        choice_index = np.random.choice(range(len(options)), p=distribution)
        options[choice_index].carry_out(hypothetical_game)
 
        child_options = [child[0] for child in self.children]
        if not options[choice_index] in child_options:
            self.children.append((options[choice_index], CFRNode(game=hypothetical_game, original_player_id=self.original_player_id, parent=self, model=self.model, training=self.training, device=self.device, depth=self.depth+1)))

            self.cumulative_regrets = np.append(self.cumulative_regrets, 0)
            self.strategy = np.append(self.strategy, 0)
            if self.model:
                winning_probabilities = self.model_inference(self.game)
                if not self.training:
                    self.pred_node_value = self.model_reward_weights * winning_probabilities
            self.cumulative_strategy = np.append(self.cumulative_strategy, 0)

    def is_terminal(self):
        return self.game.terminal

    def get_reward(self):
        return self.game.rewards

    def cfr_train(self, max_iterations=100000):
        if self.is_terminal():
            return 

        self.expand()
        
        node = self
        for i in range(max_iterations):
            # Traverse
            node.update_strategy()
            node, _ = node.action_choice()
            if node.is_terminal():
                reward = node.get_reward()
                node.backpropagate(reward)
                node.update_strategy()
                node = self
            else:
                node.expand()
        self.update_strategy()

    def cfr_pred(self, max_iterations=2000, max_depth=20):
        if self.is_terminal():
            return 
        self.expand()
        depth = 0
        node = self
        for i in range(max_iterations):
            # Traverse
            node.update_strategy()
            node, _ = node.action_choice()
            if node.depth > max_depth and not node.is_terminal():
                node.expand()
                node.backpropagate(node.pred_node_value)
                node.update_strategy()
                node = self
            elif node.is_terminal():
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
            actual_rewards = [child[1].winning_probabilities[player_id] for child in self.children]
            # Get the maximum possible reward
            max_reward = max(actual_rewards)
            # Update regrets
            sum_of_regrets_old = np.sum(self.cumulative_regrets)
            for a in range(len(self.children)):
                self.cumulative_regrets[a] += max_reward - actual_rewards[a]
                
            self.regret_gradient = np.sum(self.cumulative_regrets) - sum_of_regrets_old
            #self.total_gradients.append(self.regret_gradient)
        else:
            actual_rewards = np.array([child[1].winning_probabilities for child in self.children]).T

            max_rewards = np.max(actual_rewards, axis=0)
            regret_values = max_rewards - actual_rewards
            
            sum_of_regrets_old = np.sum(self.cumulative_regrets)
            self.cumulative_regrets += regret_values
            self.regret_gradient = np.sum(self.cumulative_regrets) - sum_of_regrets_old
            #self.total_gradients.append(self.regret_gradient)

    def get_all_targets(self, usefulness_treshold = 15):
        """
        Recursively gather the training targets from each node in the tree.

        Returns:
        - targets_list (list): List containing targets from all nodes in the tree.
        """
        targets_list = []

        # Get the targets from the current node and append to the list
        targets_list += self.build_train_targets()

        # Recursively gather targets from children nodes
        for _, child_node in self.children:
            targets_list += child_node.get_all_targets(usefulness_treshold)

        return targets_list

    def backpropagate(self, reward):
        
        # Update the value of this node
        if self.training or (self.node_value.sum() == 0 or not self.model):
            self.node_value += reward
        self.winning_probabilities = self.node_value / self.node_value.sum()
        
        # Calculate regret for this node
        if self.children:
            self.update_regrets()

        if self.parent is None:
            return
        # Recursively call backpropagate on parent node
        self.parent.backpropagate(reward)

    def update_strategy(self):

        if not self.role_pick_node:
            inverted_regrets = -self.cumulative_regrets
            transformed_regrets = np.exp(inverted_regrets * np.log(1.3))
            total_transformed_regret = np.sum(transformed_regrets)
            if total_transformed_regret > 0:  # Check to avoid division by zero
                self.strategy = transformed_regrets / total_transformed_regret
            else:
                self.strategy = np.ones_like(transformed_regrets) / len(transformed_regrets)

        else:
            inverted_regrets = -self.cumulative_regrets
            transformed_regrets = np.exp(inverted_regrets * np.log(1.3))
            # Calculate the total transformed regrets
            total_transformed_regrets = np.sum(transformed_regrets, axis=0)
            # Check if any element in total_transformed_regrets is zero or close to zero
            if np.any(total_transformed_regrets <= 1e-8):
                self.strategy = np.where(total_transformed_regrets > 1e-8,
                                         transformed_regrets / total_transformed_regrets,
                                         1.0 / transformed_regrets.shape[0])
            else:
                self.strategy = transformed_regrets / total_transformed_regrets



        self.cumulative_strategy += self.strategy
        self.cumulative_strategy = self.cumulative_strategy / self.cumulative_strategy.sum()

    def build_train_targets(self, usefulness_treshold = 15):
        if len(self.children) == 0 or self.node_value.sum() < usefulness_treshold:
            return []
        if self.role_pick_node:
            i = randint(0, 5)
            hypothetical_game = deepcopy(self.game)
            hypothetical_game.gamestate.player_id = i
            
            options_input = torch.cat([option.encode_option() for option, _ in self.children], dim=0).unsqueeze(0)
            model_input = hypothetical_game.encode_game()
            target_node_value = torch.tensor(self.node_value)
            target_decision_dist = torch.tensor(self.cumulative_regrets[i])
            if torch.sum(target_decision_dist) == 0:
                target_decision_dist = torch.ones_like(target_decision_dist)
            
            return [(model_input, options_input, target_node_value, target_decision_dist)]
        else:
            options_input = torch.cat([option.encode_option() for option, _ in self.children], dim=0).unsqueeze(0)
            model_input = self.game.encode_game()
            target_node_value = torch.tensor(self.node_value)
            target_decision_dist = torch.tensor(self.cumulative_regrets)
            if torch.sum(target_decision_dist) == 0:
                target_decision_dist = torch.ones_like(target_decision_dist)
                
            return [(model_input, options_input, target_node_value, target_decision_dist)]

    def model_inference_trans(self, game, options=None):
        if options is None:
            options_input = torch.cat([option.encode_option() for option, _ in self.children], dim=0).unsqueeze(0).to(self.device)
        else:
            options_input = torch.cat([option.encode_option() for option in options], dim=0).unsqueeze(0).to(self.device)
        model_input = game.encode_game().unsqueeze(0).to(self.device)
        
        if self.device != "cpu" and torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() < 500 * 1024**2: # 1000 MB
            raise RanOutOfMemory
        
        with torch.no_grad():
            distribution, node_value = self.model(model_input, options_input)

        distribution = square_and_normalize(distribution, dim=1).squeeze(0).detach().cpu().numpy()
        winning_probabilities = square_and_normalize(node_value, dim=1).squeeze(0).detach().cpu().numpy()
        return distribution, winning_probabilities
    
    def model_inference(self, game, options=None):
        model_input = game.encode_game().unsqueeze(0).to(self.device)
        
        if self.device != "cpu" and torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() < 500 * 1024**2: # 1000 MB
            raise RanOutOfMemory
        
        with torch.no_grad():
            node_value = self.model(model_input)

        winning_probabilities = square_and_normalize(node_value, dim=1).squeeze(0).detach().cpu().numpy()
        return winning_probabilities