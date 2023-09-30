import numpy as np
from copy import deepcopy

from algorithms.model_utils import create_mask, get_distribution, build_targets
from algorithms.model_utils import TargetBuildParams, augment_game
import torch
from game.config import role_to_role_id

class CFRNode:
    def __init__(self, game, original_player_id, parent=None, player_count = 6, role_pick_node=False, model=None):
        self.game = game # Unkown informations are present but counted as dont care
        self.model = model
        self.parent = parent
        self.children = [] # (Option that carries the game to the node, NODE)
        self.current_player_id = game.gamestate.player_id
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
            if self.cumulative_strategy.sum() == 0:
                normalized_strategy = np.ones(len(self.children)) / len(self.children)
            else:
                normalized_strategy = self.cumulative_strategy / self.cumulative_strategy.sum()

            # Choose an action according to the normalized strategy
            choice_index = np.random.choice(range(len(self.children)), p=normalized_strategy)

            # Return the chosen child and option
            return self.children[choice_index][1], self.children[choice_index][0]

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
        for i in range(max_repeat_count):
            hypothetical_game = deepcopy(self.game)
            while hypothetical_game.gamestate.state != 1:
                options, masks = hypothetical_game.get_options_from_state()

                if self.model:
                    model_input = hypothetical_game.encode_game().unsqueeze(0)
                    output = self.model(model_input).squeeze(0)
                    distribution, options_list = get_distribution(output, masks, options)
                else:
                    options_list = [option for option_list in options.values() for option in option_list]
                    distribution = np.ones(len(options_list)) / len(options_list)

                choice_index = np.random.choice(range(len(options_list)), p=distribution)
                if hypothetical_game.gamestate.player_id == self.original_player_id:
                    option_to_carry_out = options_list[choice_index]
                options_list[choice_index].carry_out(hypothetical_game)
            # Must be at the end of rolepick
            assert hypothetical_game.gamestate.player_id == hypothetical_game.get_player_from_role_id(hypothetical_game.used_roles[0]).id
            self.children.append((option_to_carry_out, CFRNode(game=hypothetical_game, original_player_id=self.original_player_id, parent=self, role_pick_node=False, model=self.model)))
        
            self.cumulative_regrets = np.zeros((1, 6)) if self.cumulative_regrets.size == 0 else np.concatenate((self.cumulative_regrets, np.zeros((1, 6))), axis=0)
            self.strategy = np.zeros((1, 6)) if self.strategy.size == 0 else np.concatenate((self.strategy, np.zeros((1, 6))), axis=0)
            self.cumulative_strategy = np.zeros((1, 6)) if self.cumulative_strategy.size == 0 else np.concatenate((self.cumulative_strategy, np.zeros((1, 6))), axis=0)
        
        self.cumulative_regrets = self.cumulative_regrets.T
        self.strategy = self.strategy.T
        self.cumulative_strategy = self.cumulative_strategy.T
        # Role pick target mask is a full mask
        self.target_params = TargetBuildParams(model_masks=[[create_mask(self.game.game_model_output_size, 6, 14, type="top_level_direct")]], options=None)
    
    def expand_for_original_player(self):
        options, masks = self.game.get_options_from_state()
        options_list = [option for option_list in options.values() for option in option_list]
        self.target_params = TargetBuildParams(model_masks=deepcopy(masks), options=options)
        for option in options_list:

            hypothetical_game = deepcopy(self.game)

            # Sample if not the same players turn as before
            if self.parent is None or hypothetical_game.gamestate.player_id != self.parent.game.gamestate.player_id:
                hypothetical_game.sample_private_information(hypothetical_game.players[self.original_player_id], role_sample=self.parent.game.gamestate.state != 0 if self.parent else False)
            option.carry_out(hypothetical_game)
            
            #print("Added child info set from orig child player's role: ", hypothetical_game.players[hypothetical_game.gamestate.player_id].role, " ID: ", hypothetical_game.gamestate.player_id, "Action leading there: ", option.name)
            self.children.append((option, CFRNode(game=hypothetical_game, original_player_id=self.original_player_id, parent=self, role_pick_node=hypothetical_game.gamestate.state == 0, model=self.model)))

        if self.model:
            model_input = self.game.encode_game().unsqueeze(0)
            output = self.model(model_input).squeeze(0)
            distribution, _ = get_distribution(output, masks, options)
        self.cumulative_regrets = np.zeros(len(self.children))
        self.strategy = np.zeros(len(self.children))
        self.cumulative_strategy = distribution if self.model else np.zeros(len(self.children))


    def expand_for_opponents(self):

        self.target_params = None
        hypothetical_game = deepcopy(self.game)
        # Sample if not the same players turn as before
        if self.parent is None or hypothetical_game.gamestate.player_id != self.parent.game.gamestate.player_id:
            hypothetical_game.sample_private_information(hypothetical_game.players[self.original_player_id], role_sample=self.parent.game.gamestate.state != 0 if self.parent else False)

        options, masks = hypothetical_game.get_options_from_state()
        if self.model:
            model_input = hypothetical_game.encode_game().unsqueeze(0)
            output = self.model(model_input).squeeze(0)
            distribution, options_list = get_distribution(output, masks, options)
        else:
            options_list = [option for option_list in options.values() for option in option_list]
            distribution = np.ones(len(options_list)) / len(options_list)

        # If no only 1 option, choose that one, mask is empty
        if distribution.size:
            choice_index = np.random.choice(range(len(options_list)), p=distribution)
        else:
            choice_index = 0
        options_list[choice_index].carry_out(hypothetical_game)
        #print("Added child info set from opponent child player's role: ", hypothetical_game.players[hypothetical_game.gamestate.player_id].role, " ID: ", hypothetical_game.gamestate.player_id, "Action leading there: ", options_list[choice_index].name)

        child_options = [child[0] for child in self.children]
        if not options_list[choice_index] in child_options:
            self.children.append((options_list[choice_index], CFRNode(game=hypothetical_game, original_player_id=self.original_player_id, parent=self, role_pick_node=hypothetical_game.gamestate.state == 0, model=self.model)))

            self.cumulative_regrets = np.append(self.cumulative_regrets, 0)
            self.strategy = np.append(self.strategy, 0)
            self.cumulative_strategy = np.append(self.cumulative_strategy, 0)


    def is_terminal(self):
        return self.game.terminal

    def get_reward(self):
        return self.game.rewards

    def cfr(self, max_iterations=100000):
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
                pass
                #print(f"cfr{i}, Traversion: ", action.name, "choice: ", action.attributes["role"])
            else:
                pass
                #print(f"cfr{i}, Traversion: ", action.name)
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
    
    def get_all_targets(self):
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
            targets_list += child_node.get_all_targets()

        return targets_list

    
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

    def set_role_favorabilities(self):
        """
        Creating a favorability object that is used
        to create target policys for the rolepick phase
        """

        # Initialize a 6x8 matrix filled with zeros
        role_favorability = np.zeros((6, 8))

        # Iterate over the 10 children of the node
        for i in range(len(self.children)):
            child = self.children[i][1]

            # Get the strategy values for this child
            strategy_values = self.strategy[:, i]

            # Update the role favorability matrix based on this child's strategy values
            for player_id in range(6):
                role_id = role_to_role_id[child.game.players[player_id].role]
                role_favorability[player_id, role_id] += strategy_values[player_id]
        
        row_sums = role_favorability.sum(axis=1)[:, np.newaxis]
        if row_sums.sum() == 0:
            normalized_role_favorability = np.ones((6, 8)) / 8
        else:
            normalized_role_favorability = role_favorability / row_sums

        self.role_favorability = normalized_role_favorability


    def update_strategy(self):
        self.cumulative_strategy += self.strategy
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
            self.set_role_favorabilities()

        # Update the cumulative strategy used for the average strategy output


    def build_train_targets(self):
        if self.children and self.target_params:
            if self.role_pick_node:
                full_targets = []
                for i in range(len(self.game.players)):
                    self.game.gamestate.player_id = i
                    targets, loss_mask = build_targets(self.target_params.model_masks, self.role_favorability[i], self.node_value)
                    full_targets.append([self.game.encode_game(), targets, loss_mask])
                return full_targets
            else:
                targets, loss_mask = build_targets(self.target_params.model_masks, self.strategy, self.node_value, options=self.target_params.options)
                game, loss_mask, targets = augment_game(self.game,loss_mask, targets)
                return [[game.encode_game(), targets, loss_mask]]
        else:
            return []