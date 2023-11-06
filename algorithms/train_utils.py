
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

def draw_eval_results(results, base_usefullness_treshold, max_usefullness_theshold, name="loss_plot.png"):
    x_values = list(range(base_usefullness_treshold, max_usefullness_theshold))
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.lineplot(x=x_values, y=results)
    plt.xlabel("Usefulness Treshold")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss vs Usefulness Treshold")
    plt.savefig(name)
    plt.close()

class RanOutOfMemory(Exception):
    pass

def draw_length_results(lengths, base_usefullness_treshold, max_usefullness_theshold, name="length_plot.png"):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.lineplot(x=list(range(base_usefullness_treshold, max_usefullness_theshold)), y=lengths)
    plt.xlabel("Usefulness Treshold")
    plt.ylabel("Number of Targets")
    plt.title("Number of Targets vs Usefulness Treshold")
    plt.savefig(name)
    plt.close()


def get_nodes_with_usefulness_treshold(targets, treshold):
    """
    args:
        targets: list of targets
        treshold: the minimum usefulness of a target
    returns:
        list of targets with usefulness >= treshold
    """
    return [target for target in targets if target[3].sum() >= treshold]

def get_average_regrets(targets):
    """
    args:
        targets: list of targets
    returns:
        list of average regrets
    """
    return [target[3].mean() for target in targets]


def calculate_means(list_of_lists):
    # Sum each 3rd and 4th element tensor
    summed_third_elements = torch.tensor([int(lst[2].sum()) for lst in list_of_lists])
    summed_fourth_elements = torch.tensor([lst[3].sum() for lst in list_of_lists])

    # Find the maximum sum among the 3rd elements to set as the max threshold
    max_sum = summed_third_elements.max().item()

    # Calculate the means for each threshold
    means = []
    thresholds = range(max_sum, -1, -1)

    for threshold in thresholds:
        # Select the summed 4th elements where the corresponding 3rd element sum is above the threshold
        valid_sums = summed_fourth_elements[summed_third_elements >= threshold]
        # Calculate the mean of these valid sums
        mean_value = valid_sums.mean().item() if len(valid_sums) > 0 else 0
        means.append(mean_value)

    return list(thresholds), means

# This function will plot the data using seaborn
def plot_threshold_means(thresholds, means, name="threshold_means.png"):
    sns.lineplot(x=thresholds, y=means)
    plt.xlabel('Threshold')
    plt.ylabel('Mean of Sums')
    plt.title('Mean of the 4th Element Sums by Threshold')
    plt.savefig(name)
    

def plot_avg_regrets(targets, name="avg_regrets.png"):
    # Calculate means
    thresholds, means = calculate_means(targets)

    # Plot the results
    plot_threshold_means(thresholds, means, name=name)


def square_and_normalize(input_tensor, dim=-1):
    squared = torch.pow(input_tensor, 2)
    return squared / squared.sum(dim=dim, keepdim=True)


def log_square_and_normalize(model_outputs):
    # Squaring and normalizing the model outputs
    
    normalized_outputs = square_and_normalize(model_outputs)
    log_normalized_outputs = torch.log(normalized_outputs + 1e-10)  # small value to prevent log(0)

    return log_normalized_outputs
