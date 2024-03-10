import pickle as pkl
import logging
from algorithms.train import train_node_value_only
logging.basicConfig(level=logging.INFO)


with open(f"10k_50thresh_pretrain.pkl", 'rb') as file:
    targets = pkl.load(file)

with open(f"validation_targets.pkl", 'rb') as file:
    val_targets = pkl.load(file)


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(params, trial_id):
    # Unpack your parameters
    lr, hidden_size, gamma = params['lr'], params['hidden_size'], params['gamma']

    # Call your training function
    loss = train_node_value_only(targets, val_targets, lr=lr, hidden_size=hidden_size, gamma=gamma, epochs=1000, parent_folder=f"pretrain_{trial_id}", batch_size=512, verbose=False)
    
    # Hyperopt tries to minimize the objective, so return the loss
    return {'loss': loss, 'status': STATUS_OK}

# Define the search space for hyperparameters
space = {
    'lr': hp.loguniform('lr', -5, 0), # for learning rate, a log scale is often helpful
    'hidden_size': hp.choice('hidden_size', [128, 256, 512, 1024]),
    'gamma': hp.uniform('gamma', 0.5, 1)
}

# Run the optimization
trials = Trials()
best = fmin(fn=lambda params: objective(params, trial_id=trials.trials[-1]['tid']), 
            space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best Hyperparameters:")
print(best)

logging.shutdown()