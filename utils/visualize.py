import os
import numpy as np
import matplotlib.pyplot as plt


def is_pareto_optimal(points):
    """
    Finds the Pareto optimal points from a set of 2D points.
    Assumes we want to MAXIMIZE both objectives.

    Args:
        points (np.ndarray): An (n_points, 2) array of objective values.

    Returns:
        np.ndarray: A boolean array of shape (n_points,) where True
                    indicates the point is on the Pareto frontier.
    """
    n_points = points.shape[0]
    is_optimal = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        # Check if point i is dominated by any other point
        for j in range(n_points):
            if i == j:
                continue
            
            # A point j dominates point i if:
            # 1. It's at least as good in all objectives: (points[j] >= points[i]).all()
            # 2. It's strictly better in at least one objective: (points[j] > points[i]).any()
            if (points[j] >= points[i]).all() and (points[j] > points[i]).any():
                is_optimal[i] = False
                break # Point i is dominated, no need to check further

    return is_optimal


def plot_eval_pareto(eval_results, log_dir, env_name, dataset, iter=0, train_returns=None):

    x, y, all_returns, prefs = [], [], [], []
    for ep, value in eval_results.items():
        preferences = value['preferences']
        returns = value['raw_returns']
        # lengths = value['lengths']
        # weighted_returns = value['weighted returns']
        x.append(returns[0])
        y.append(returns[1])
        all_returns.append(returns)
        prefs.append(preferences)

    all_returns = np.array(all_returns)

    pareto_mask = is_pareto_optimal(all_returns)
    pareto_points = all_returns[pareto_mask]
    non_pareto_points = all_returns[~pareto_mask]
    # print("\n\n\n", all_returns, "\n\n\n")
    # print("\n\n\n", prefs, "\n\n\n")
    # print("\n\n\n", pareto_mask, "\n\n\n")
    prefs = np.array(prefs)
    pareto_prefs = prefs[pareto_mask]
    non_pareto_prefs = prefs[~pareto_mask]

    pareto_info = {
        "pareto_points" : pareto_points.tolist(),
        "non_pareto_points" : non_pareto_points.tolist(),
        "pareto_prefs" : pareto_prefs.tolist(),
        "non_pareto_prefs" : non_pareto_prefs.tolist()}
        

    import json
    os.makedirs(os.path.join(log_dir, 'pareto'), exist_ok=True)
    filename = os.path.join(log_dir, 'pareto', f'eval_{all_returns.shape[0]}points_iter{iter}.json')
    with open(filename, 'w') as fp:
        json.dump(pareto_info, fp)

    


    # plt.plot(x, y, c='red', s=100, marker='*', zorder=3, label='all points')
    

    
    plt.figure(figsize=(5, 5))

    
    if train_returns is not None:
        # Plot training points
        plt.scatter(train_returns[:, 0], train_returns[:, 1], c='green', s=30, alpha=0.5, label='Training Points')

    # Plot non-dominated points (Pareto frontier)
    # plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'r-', label='Pareto Frontier', linewidth=2)
    plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='red', s=50, zorder=3, label='Non-dominated Points')

    # Plot dominated points
    plt.scatter(non_pareto_points[:, 0], non_pareto_points[:, 1], c='blue', s=30, alpha=0.5, label='Dominated Points')

    # --- Configure the Plot ---
    plt.title(f'MODT(P) : {env_name} - {dataset}', fontsize=12)
    plt.xlabel('Objective 1', fontsize=10)
    plt.ylabel('Objective 2', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot to a file
    plt.savefig(os.path.join(log_dir, 'pareto',  f'pareto_{all_returns.shape[0]}points_iter{iter}.png'))






