
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

def square_and_normalize(input_tensor, dim=-1):
    squared = torch.pow(input_tensor, 2)
    return squared / squared.sum(dim=dim, keepdim=True)


def log_square_and_normalize(model_outputs):
    # Squaring and normalizing the model outputs
    
    normalized_outputs = square_and_normalize(model_outputs)
    log_normalized_outputs = torch.log(normalized_outputs + 1e-10)  # small value to prevent log(0)

    return log_normalized_outputs
