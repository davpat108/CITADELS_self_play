
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def draw_eval_results(results, base_usefullness_treshold, max_usefullness_theshold, name="loss_plot.png"):
    sns.set(style="darkgrid") 
    x_values = list(range(base_usefullness_treshold, max_usefullness_theshold, 5))
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
    sns.set(style="darkgrid") 
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.lineplot(x=list(range(base_usefullness_treshold, max_usefullness_theshold, 5)), y=lengths)
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


def calculate_means(list_of_lists):
    """
    Method to calculate the regrets for later plotting, categorized by the amount of children a node has.
    Also using thresholds, as the minimum times the node has to be backpropagated to be considered.
    """
    # Sum each 3rd and 4th element tensor
    summed_third_elements = torch.tensor([int(lst[2].sum()) for lst in list_of_lists])
    summed_fourth_elements = torch.tensor([lst[3].sum() for lst in list_of_lists])

    # Categorize the 4th elements based on their length
    cat_2 = torch.tensor([lst[3].sum() for lst in list_of_lists if len(lst[3]) == 2])
    summed_cat_2_third_elements = torch.tensor([int(lst[2].sum()) for lst in list_of_lists if len(lst[3]) == 2])
    
    cat_3_6 = torch.tensor([lst[3].sum() for lst in list_of_lists if 3 <= len(lst[3]) <= 6])
    summed_cat_3_6_third_elements = torch.tensor([int(lst[2].sum()) for lst in list_of_lists if 3 <= len(lst[3]) <= 6])
    
    cat_7_30 = torch.tensor([lst[3].sum() for lst in list_of_lists if 7 <= len(lst[3]) <= 30])
    summed_cat_7_30_third_elements = torch.tensor([int(lst[2].sum()) for lst in list_of_lists if 7 <= len(lst[3]) <= 30])
    
    cat_30_plus = torch.tensor([lst[3].sum() for lst in list_of_lists if len(lst[3]) > 30])
    summed_cat_30_plus_third_elements = torch.tensor([int(lst[2].sum()) for lst in list_of_lists if len(lst[3]) > 30])

    # Find the maximum sum among the 3rd elements to set as the max threshold
    max_sum = summed_third_elements.max().item()

    # Calculate the means for each threshold
    means = []
    cat_2_means = []
    cat_3_6_means = []
    cat_7_30_means = []
    cat_30_plus_means = []
    
    cat_2_lengths = []
    cat_3_6_lengths = []
    cat_7_30_lengths = []
    cat_30_plus_lengths = []
    
    thresholds = range(1000, -1, -1)


    for threshold in thresholds:
        # Select the summed 4th elements where the corresponding 3rd element sum is above the threshold
        valid_sums = summed_fourth_elements[summed_third_elements >= threshold]
        # Calculate the mean of these valid sums
        mean_value = valid_sums.mean().item()
        means.append(mean_value)

        # Calculate means for each category
        cat_2_means.append(cat_2[summed_cat_2_third_elements >= threshold].mean().item() if len(cat_2) > 0 else 0)
        cat_3_6_means.append(cat_3_6[summed_cat_3_6_third_elements >= threshold].mean().item() if len(cat_3_6) > 0 else 0)
        cat_7_30_means.append(cat_7_30[summed_cat_7_30_third_elements >= threshold].mean().item() if len(cat_7_30) > 0 else 0)
        cat_30_plus_means.append(cat_30_plus[summed_cat_30_plus_third_elements >= threshold].mean().item() if len(cat_30_plus) > 0 else 0)
        
        cat_2_lengths.append(len(cat_2[summed_cat_2_third_elements >= threshold]))
        cat_3_6_lengths.append(len(cat_3_6[summed_cat_3_6_third_elements >= threshold]))
        cat_7_30_lengths.append(len(cat_7_30[summed_cat_7_30_third_elements >= threshold]))
        cat_30_plus_lengths.append(len(cat_30_plus[summed_cat_30_plus_third_elements >= threshold]))


    return thresholds, means, cat_2_means, cat_3_6_means, cat_7_30_means, cat_30_plus_means, cat_2_lengths, cat_3_6_lengths, cat_7_30_lengths, cat_30_plus_lengths

# This function will plot the data using seaborn
def plot_threshold_means(thresholds, means, cat_2_means, cat_3_6_means, cat_7_30_means, cat_30_plus_means, cat_2_lengths, cat_3_6_lengths, cat_7_30_lengths, cat_30_plus_lengths, name="threshold_means.png"):
    sns.set(style="darkgrid") 
    # Plot for the means
    plt.figure(figsize=(10, 8))
    sns.lineplot(x=thresholds, y=means, label='All')
    sns.lineplot(x=thresholds, y=cat_2_means, label='Length = 2')
    sns.lineplot(x=thresholds, y=cat_3_6_means, label='Length 3-6')
    sns.lineplot(x=thresholds, y=cat_7_30_means, label='Length 7-30')
    sns.lineplot(x=thresholds, y=cat_30_plus_means, label='Length > 30')
    plt.xlabel('Threshold')
    plt.ylabel('Mean of Sums')
    plt.title('Mean of regrets by minimum backprop limit')
    plt.legend()
    plt.savefig(name)

    # Plot for the counts
    plt.figure(figsize=(10, 8))
    sns.lineplot(x=thresholds, y=cat_2_lengths, label='Length = 2')
    sns.lineplot(x=thresholds, y=cat_3_6_lengths, label='Length 3-6')
    sns.lineplot(x=thresholds, y=cat_7_30_lengths, label='Length 7-30')
    sns.lineplot(x=thresholds, y=cat_30_plus_lengths, label='Length > 30')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.ylim(0, 500)
    plt.title('Count of Elements by Length Category')
    plt.legend()
    plt.savefig(name+"_length.png")

def plot_avg_regrets(targets, name="avg_regrets.png"):
    # Calculate means
    thresholds, means, cat_2_means, cat_3_6_means, cat_7_30_means, cat_30_plus_means, cat_2_lengths, cat_3_6_lengths, cat_7_30_lengths, cat_30_plus_lengths  = calculate_means(targets)

    # Plot the results
    plot_threshold_means(thresholds, means, cat_2_means, cat_3_6_means, cat_7_30_means, cat_30_plus_means, cat_2_lengths, cat_3_6_lengths, cat_7_30_lengths, cat_30_plus_lengths, name=name)


def square_and_normalize(input_tensor, dim=-1):
    squared = torch.pow(input_tensor, 2)
    return squared / squared.sum(dim=dim, keepdim=True)


def log_square_and_normalize(model_outputs):
    # Squaring and normalizing the model outputs
    
    normalized_outputs = square_and_normalize(model_outputs)
    log_normalized_outputs = torch.log(normalized_outputs + 1e-10)  # small value to prevent log(0)

    return log_normalized_outputs
