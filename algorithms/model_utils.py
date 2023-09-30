import numpy as np
import torch
from itertools import product
import torch.nn.functional as F
from random import shuffle
from copy import deepcopy

class Mask():
    def __init__(self, type:str="top_level", model_size:int=1521) -> None:
        """
        Types: "top_level", "direct", "deck", "warlord", "unrepresented", "empty"
        top_level: top level decision with or without a followup decision
        direct: direct decision after top level decision
        deck: deck decision after top level decision
        warlord: warlord (6 deck merged) decision after top level decision
        unrepresented: top level decision with a followup decision that is not represented by the model output, like seer
        top_level_direct: top level decision that has no other decisons with it, like rolepick
        empty: ignored decision
        """
        self.mask = np.zeros(model_size, dtype=int)
        self.type = type
    
    def set_mask(self, start_index:int, end_index:int, pick_indexes:list=None) -> None:
        if pick_indexes is None:
            self.mask[start_index:end_index] = 1
        else:
            self.mask[start_index:end_index][pick_indexes] = 1

        self.start_index = start_index
        self.end_index = end_index

class TargetBuildParams():
    def __init__(self, model_masks:list, options:dict) -> None:
        """
        Information build target function needs to create the expected model output
        """
        self.model_masks = model_masks
        self.options = options

def augment_game(game, mask, targets):
    """
    Shuffles the players around so the model wont overfit to the original player
    """
    new_game = deepcopy(game)
    new_mask = deepcopy(mask)
    new_targets = deepcopy(targets)

    shuffled_player_order = [0,1,2,3,4,5]
    shuffle(shuffled_player_order)
    for i in range(len(new_game.players)):
        new_game.players[i].id = shuffled_player_order[i]
    new_game.players.sort(key=lambda x: x.id)
    #Node value

    new_targets[0:6] = new_targets[0:6][shuffled_player_order]
    #Magician
    new_mask[170:176] = new_mask[170:176][shuffled_player_order]
    new_targets[170:176] = new_targets[170:176][shuffled_player_order]
    #Wizard
    new_mask[218:224] = new_mask[218:224][shuffled_player_order]
    new_targets[218:224] = new_targets[218:224][shuffled_player_order]

    return new_game, new_mask, new_targets



def create_mask(model_output_size, start_index, end_index, pick_indexes=None, type="top_level"):
    """
    Create a mask vector based on the given pick indexes within a specified interval.
    
    Args:
    - flattened_output (numpy.ndarray): Output size of the model
    - start_index (int): The starting index of the interval.
    - end_index (int): The ending index of the interval.
    - pick_indexes (list, optional): List of indexes to pick within the interval.
    Returns:
    - Mask: A mask vector of the same length as the flattened output, with ones at the specified pick indexes within the interval and zeros elsewhere.
    """

    mask = Mask(type=type, model_size=model_output_size)
    mask.set_mask(start_index, end_index, pick_indexes)
    return mask

#def create_mask(model_output_size, start_index, end_index, pick_indexes=None):
#    """
#    Create a mask vector based on the given pick indexes within a specified interval.
#    
#    Args:
#    - flattened_output (numpy.ndarray): Output size of the model
#    - start_index (int): The starting index of the interval.
#    - end_index (int): The ending index of the interval.
#    - pick_indexes (list, optional): List of indexes to pick within the interval.
#    Returns:
#    - numpy.ndarray: A mask vector of the same length as the flattened output, with ones at the specified pick indexes within the interval and zeros elsewhere.
#    """
#
#    mask = np.zeros(model_output_size, dtype=int)
#    if pick_indexes is None:
#        mask[start_index:end_index] = 1
#    else:
#        mask[start_index:end_index][pick_indexes] = 1
#    return mask

def get_distribtuion_from_deck(mask, nn_output):
    """
    Get the probability distribution for choices of cards based on the neural network output.
    Args:
    - mask (torch.Tensor): A binary mask indicating which cards are present.
    - nn_output (torch.Tensor): The neural network output for the cards.
    Returns:
    - torch.Tensor: A probability distribution for choices of cards.
    """

    # Get the indices of the cards present in the hand
    card_indices = torch.nonzero(mask).squeeze()
    # Extract the relevant scores from the neural network output
    card_scores = nn_output[card_indices]
    # Convert the scores to a probability distribution
    probabilities = card_scores / card_scores.sum()

    if probabilities.dim() == 0:
        probabilities = probabilities.unsqueeze(0)

    return probabilities

def get_combined_deck_choices_for_warlords(nn_output, mask, num_decks=6, deck_size=40, start_index=0):


    deck_distributions = []
    # Iterate over each deck
    for i in range(num_decks):
        start_idx = i * deck_size + start_index
        end_idx = (i + 1) * deck_size + start_index

        # Create a mask that only has ones between start_idx and end_idx
        current_mask = torch.zeros_like(mask)
        current_mask[start_idx:end_idx] = 1
        effective_mask = mask * current_mask

        # Get the distribution for the current deck
        deck_distribution = get_distribtuion_from_deck(effective_mask, nn_output)
        deck_distributions.append(deck_distribution)

    # Compute the joint distribution over all decks
    # Filter out empty distributions
    final_distribution = torch.cat(deck_distributions)
    final_distribution /= final_distribution.sum()

    return final_distribution



def get_distribution(model_output, model_masks, options):
    """
    Compute a conditional probability distribution for a set of options based on a hierarchical decision-making process.
    The function uses the top-level masks to get a probability distribution for the top-level decisions. Then based on the name of the masks
    builds a distribution

    Parameters:
    - model_output (torch.Tensor): The output of the model, a vector of size .
    - model_masks (list of lists of masks): A hierarchical list of masks. Types: "top_level", "top_level_direct" "direct", "deck",  "unrepresented", "empty"
    - options (dict): A dictionary where keys correspond to top-level decisions and values are lists of options 
      available for each top-level decision.
    Returns:
    - np.array: A probability distribution over all the options, taking into account the hierarchical decision-making process.
    - list: all the options in the same order as the probability distribution
    Example:
    model_output = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7])
    model_masks = [
        [mask.mask:torch.tensor([True, True, False, False, False, False, False, False]),
         mask.mask:torch.tensor([False, False, True, True, False, False, False, False])],
        [mask.mask:torch.tensor([False, False, False, False, True, True, False, False])],
        [mask.mask:torch.tensor([False, False, False, False, False, False, True, True])]
    ]
    options = {0: ["option1", "option2"], 1: ["option3"], 2: ["option4", "option5"]}
    distribution = get_distribution(model_output, model_masks, options)
    """
    final_probs = []

    top_level_bits = torch.stack([torch.tensor(mask[0].mask, dtype=torch.bool) for mask in model_masks])
    scores = model_output[torch.any(top_level_bits, dim=0)]
    top_level_probs = F.softmax(scores, dim=0)
    top_index = 0

    for i, masks in enumerate(model_masks):
        if i in options:
            # If there are subsequent decisions and the model output represents them, use the model output
            if masks[0].type == "top_level":
                if len(masks) == 1:
                    subsequent_probs = torch.ones(1)
                elif masks[1].type == "deck":
                    subsequent_bits = torch.tensor(masks[1].mask, dtype=torch.bool)
                    subsequent_probs = get_distribtuion_from_deck(mask=subsequent_bits, nn_output=model_output)

                elif masks[1].type == "direct":
                    subsequent_bits = torch.tensor(masks[1].mask, dtype=torch.bool)
                    subsequent_probs = F.softmax(model_output[subsequent_bits], dim=0)
                elif masks[1].type == "unrepresented":
                    # If not represented by the model output, just use uniform distribution
                    subsequent_probs = torch.ones(len(options[i]))/len(options[i])
                else:
                    raise Exception("Unknown mask combination")


                combined_probs = top_level_probs[top_index] * subsequent_probs
                top_index += 1
                # Sometimes combined_probs is just a sclar, sometimes it's a vector
                final_probs.extend([combined_probs.item()] if combined_probs.dim() == 0 else combined_probs.tolist())

            # if there are subsequent decisions but the model output doesn't represent them, just use uniform distribution
            # This should catch seer
            elif masks[0].type == "unrepresented":
                # If not represented by the model output, just use uniform distribution
                subsequent_probs = torch.ones(len(options[i]))/len(options[i])
                combined_probs = top_level_probs[top_index] * subsequent_probs
                top_index += 1

                # Sometimes combined_probs is just a sclar, sometimes it's a vector
                final_probs.extend([combined_probs.item()] if combined_probs.dim() == 0 else combined_probs.tolist())

            # This should catch rolepick
            elif masks[0].type == "top_level_direct" or masks[0].type == "empty":
                final_probs = top_level_probs.tolist()
            
            else:
                raise Exception("Unknown mask combination")


    
    final_probs = np.array(final_probs)
    final_probs /= final_probs.sum()
    assert len(final_probs) == len([option for option_list in options.values() for option in option_list])
    return final_probs, [option for option_list in options.values() for option in option_list]

def get_deck_targets(mask, distribution):

    # Initialize the target array for the deck with zeros
    deck_target = np.zeros(40, dtype=np.float32)

    # Get the indices of the cards present in the hand
    card_indices = np.nonzero(mask.mask)[0]
    card_indices -= mask.start_index

    # Assign the probabilities from the distribution to the corresponding positions in the target array
    deck_target[card_indices] = distribution

    return deck_target


def combine_masks(model_masks, options):

    combined_mask = np.zeros_like(model_masks[0][0].mask, dtype=bool)
    for i, masks in enumerate(model_masks):
        if i in options:
            for mask in masks:
                combined_mask = np.logical_or(combined_mask, mask.mask)
    combined_mask[0:6] = True
    return combined_mask


def build_targets(model_masks, distribution, node_value, options={0:[]}):
    """
    Reconstruct the target tensor for model training based on the given mask and distribution. 
    Inverse of get_distribution.

    Args:
    - model_masks (list of lists of masks): A hierarchical list of masks. Types: "top_level", "top_level_direct", "direct", "deck", "empty"
    - distribution (torch.Tensor): The probability distribution over all the options.
    - winning_probabilities (torch.Tensor): The winning probabilities for each option.

    Returns:
    - torch.Tensor: The reconstructed target tensor for model training.
    """
    
    target = np.zeros_like(model_masks[0][0].mask, dtype=np.float32)
    dist_index = 0


    for i, masks in enumerate(model_masks):
        if i in options:
            if masks[0].type == "top_level":

                if len(masks) > 1:
                    target[masks[1].mask.astype(bool)] = distribution[dist_index:dist_index+masks[1].mask.sum()]
                    target[masks[1].mask.astype(bool)] /= target[masks[1].mask.astype(bool)].sum()

                    target[masks[0].start_index] = distribution[dist_index:dist_index+masks[1].mask.sum()].sum()
                    dist_index += masks[1].mask.sum()

                else:
                    target[masks[0].start_index] = distribution[dist_index:dist_index+len(options[i])].sum()
                    dist_index += len(options[i])


            elif masks[0].type == "top_level_direct":
                target[masks[0].mask.astype(bool)] = distribution

    node_value_torch = torch.from_numpy(node_value)
    if node_value_torch.sum() == 0:
        node_value_torch = torch.ones_like(node_value_torch)

    node_value_torch /= node_value_torch.sum()
    
    target = torch.from_numpy(target)
    target[0:6] = node_value_torch
    combined_masks = torch.from_numpy(combine_masks(model_masks, options))

    return target, combined_masks
