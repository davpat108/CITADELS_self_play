import numpy as np
import torch
from itertools import product
import torch.nn.functional as F


def create_mask(model_output_size, start_index, end_index, pick_indexes=None):
    """
    Create a mask vector based on the given pick indexes within a specified interval.
    
    Args:
    - flattened_output (numpy.ndarray): Output size of the model
    - start_index (int): The starting index of the interval.
    - end_index (int): The ending index of the interval.
    - pick_indexes (list, optional): List of indexes to pick within the interval.
    Returns:
    - numpy.ndarray: A mask vector of the same length as the flattened output, with ones at the specified pick indexes within the interval and zeros elsewhere.
    """

    mask = np.zeros(model_output_size, dtype=int)
    if pick_indexes is None:
        mask[start_index:end_index] = 1
    else:
        mask[start_index:end_index][pick_indexes] = 1
    return mask

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

    return probabilities

def get_combined_deck_choices_for_warlords(nn_output, mask, num_decks=6, deck_size=40):
    """
    Get the combined choices for all decks based on the neural network output.
    Args:
    - nn_output (torch.Tensor): The neural network output for the cards, shaped (6*40,).
    - mask (torch.Tensor): A binary mask indicating which cards are present, shaped (6*40,).
    Returns:
    - torch.Tensor: A joint probability distribution over all possible combinations of choices.
    """

    deck_distributions = []
    # Iterate over each deck
    for i in range(num_decks):
        start_idx = i * deck_size
        end_idx = (i + 1) * deck_size
        # Extract the scores and mask for the current deck
        deck_scores = nn_output[start_idx:end_idx]
        deck_mask = mask[start_idx:end_idx]
        # Get the distribution for the current deck
        deck_distribution = get_distribtuion_from_deck(deck_mask, deck_scores)
        deck_distributions.append(deck_distribution)
    # Compute the joint distribution over all decks
    joint_distribution = torch.tensor(list(product(*deck_distributions)))
    joint_distribution = joint_distribution.prod(dim=1)
    joint_distribution /= joint_distribution.sum()
    return joint_distribution


def get_distribution(model_output, model_masks, options):
    """
    Compute a conditional probability distribution for a set of options based on a hierarchical decision-making process.
    The function uses the top-level masks to get a probability distribution for the top-level decisions. For each 
    top-level decision, if there are subsequent decisions, it uses the corresponding masks to get a probability 
    distribution conditioned on the top-level decision. The final output is a single vector representing the 
    conditional probability distribution for all options.

    Parameters:
    - model_output (torch.Tensor): The output of the model, a vector of size (1487 currently).
    - model_masks (list of lists of torch.Tensor): A hierarchical list of masks. Each top-level list corresponds to 
      a top-level decision, and each subsequent list corresponds to subsequent decisions conditioned on the top-level 
      decision.
    - options (dict): A dictionary where keys correspond to top-level decisions and values are lists of options 
      available for each top-level decision.
    Returns:
    - list: A probability distribution over all the options, taking into account the hierarchical decision-making process.
    - list: all the options in the same order as the probability distribution
    Example:
    model_output = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7])
    model_bits = [
        [torch.tensor([True, True, False, False, False, False, False, False]),
         torch.tensor([False, False, True, True, False, False, False, False])],
        [torch.tensor([False, False, False, False, True, True, False, False])],
        [torch.tensor([False, False, False, False, False, False, True, True])]
    ]
    options = {0: ["option1", "option2"], 1: ["option3"], 2: ["option4", "option5"]}
    distribution = get_distribution(model_output, model_bits, options)
    """
    final_probs = []

    for i, masks in enumerate(model_masks):

        top_level_bits = model_masks[i][0]
        top_level_probs = F.softmax(model_output[top_level_bits], dim=0)

        if len(masks) > 1 and i in options:
            for j, option in enumerate(options[i]):
                subsequent_bits = model_masks[i][1]
                if len(model_masks[i][1]) == 40:
                    subsequent_probs = get_distribtuion_from_deck(mask=subsequent_bits, nn_output=model_output)
                elif len(model_masks[i][1]) == 240:
                    subsequent_probs = get_combined_deck_choices_for_warlords(nn_output=model_output, mask=subsequent_bits)
                else:
                    subsequent_probs = F.softmax(model_output[subsequent_bits], dim=0)


                combined_probs = top_level_probs[j] * subsequent_probs
                final_probs.extend(combined_probs.tolist())
        elif i in options and len(options[i]) > 1:
            # If not represented by the model output, just use uniform distribution
            subsequent_probs = torch.ones(options[i])/len(options[i])
            combined_probs = top_level_probs[j] * subsequent_probs
            final_probs.extend(combined_probs.tolist())

        else:
            final_probs.extend(top_level_probs.tolist())
    return final_probs, [option for option_list in options.values() for option in option_list]

def get_deck_probs(mask, distribution):
    """
    Reconstruct the neural network output for choices of cards based on the given probability distribution.
    Args:
    - mask (torch.Tensor): A binary mask indicating which cards are present.
    - distribution (torch.Tensor): The probability distribution for choices of cards.
    Returns:
    - torch.Tensor: Reconstructed neural network output for the cards.
    """

    card_indices = torch.nonzero(mask).squeeze()
    total_score = distribution.sum()

    card_scores = distribution * total_score
    nn_output = torch.zeros_like(mask, dtype=torch.float32)

    nn_output[card_indices] = card_scores

    return nn_output

def get_combined_probs_warlords(mask, distribution, num_decks=6, deck_size=40):
    """
    Reconstruct the neural network output for choices of cards in multiple decks based on the given joint distribution.
    Args:
    - joint_distribution (torch.Tensor): The joint probability distribution over all possible combinations of choices.
    - mask (torch.Tensor): A binary mask indicating which cards are present, shaped (6*40,).
    Returns:
    - torch.Tensor: Reconstructed neural network output for the cards in multiple decks.
    """
    deck_distributions = []
    for i in range(num_decks):
        start_idx = i * deck_size
        end_idx = (i + 1) * deck_size
        deck_mask = mask[start_idx:end_idx]
        num_valid_cards = deck_mask.sum().item()
        deck_distribution = distribution.view(-1, num_valid_cards).sum(dim=0)
        deck_distribution /= deck_distribution.sum()
        deck_distributions.append(deck_distribution)


    nn_outputs = []
    for i, deck_distribution in enumerate(deck_distributions):
        start_idx = i * deck_size
        end_idx = (i + 1) * deck_size
        deck_mask = mask[start_idx:end_idx]
        deck_nn_output = get_deck_probs(deck_mask, deck_distribution)
        nn_outputs.append(deck_nn_output)
    taget_output = torch.cat(nn_outputs)
    return taget_output

def build_targets(model_masks, distribution, winning_probabilities):
    """
    Reconstruct the target tensor for model training based on the given mask and distribution. Inverse of get_distribution.
    Parameters:
    - mask (torch.Tensor): A binary mask indicating which bits of the model output are relevant.
    - distribution (list): A probability distribution over the options.
    Returns:
    - torch.Tensor: A target tensor for model training.
    """
    # Initialize the target tensor with zeros
    target = torch.zeros_like(model_masks, dtype=torch.float32)
    target[:winning_probabilities.shape[0]] = winning_probabilities
    dist_idx = 0
    for i, masks in enumerate(model_masks):
        top_level_bits = model_masks[i][0]
        if len(masks) > 1:

            num_subsequent_options = len(masks[1])
            top_level_probs = sum(distribution[dist_idx:dist_idx+num_subsequent_options])
            target[top_level_bits] = top_level_probs
            dist_idx += num_subsequent_options

            subsequent_bits = model_masks[i][1]
            if len(subsequent_bits) == 40:
                target[subsequent_bits] = get_deck_probs(subsequent_bits[dist_idx:dist_idx+40], distribution[dist_idx:dist_idx+40])
                dist_idx += 40
            elif len(subsequent_bits) == 240:
                target[subsequent_bits] = get_combined_probs_warlords(subsequent_bits[dist_idx:dist_idx+240], distribution[dist_idx:dist_idx+240])
                dist_idx += 240
            else:
                target[subsequent_bits] = distribution[dist_idx:dist_idx+num_subsequent_options]
                dist_idx += num_subsequent_options

        else:
            target[top_level_bits] = distribution[dist_idx]
            dist_idx += 1
    return target