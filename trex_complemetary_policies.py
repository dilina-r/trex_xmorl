import os
import argparse
import pickle
import numpy as np
import json
import gym
from gym.spaces.box import Box
import environments
import torch
from torch.optim import AdamW as Optimizer
from pprint import pprint
from utils.visualize import plot_eval_pareto
os.environ["CUDA_VISIBLE_DEVICES"] = "1"




def run(config):

    ## Create real env
    env_name = config['env_name']
    print(f"Env name: {env_name}")
    env = gym.make(env_name)

    device = config['device']
    act_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    reward_size = env.obj_dim
    mo_rtg = True
    pref_dim = reward_size
    state_dim += pref_dim * config['concat_state_pref']
    rtg_dim = pref_dim if mo_rtg else 1

    ## Update config
    config['state_dim'] = state_dim
    config['act_dim'] = act_dim
    config['pref_dim'] = pref_dim
    config['rtg_dim'] = rtg_dim

    cluster_id = config['xmorl']['cluster_id']
    if cluster_id is None:
        log_dir = os.path.join(config['dir'], 'original')
    else:
        log_dir = os.path.join(config['dir'], f'c{cluster_id}')
    
    
    n_inner = 4 * config["hidden_size"]
    
    

    from models.modt.decision_transformer import DecisionTransformer
    model = DecisionTransformer(
        n_inner=n_inner,
        n_positions=1024,
        **config
    ).to(device)

    # exit()

    # from dataloader.morl.offline_loader import OfflineLoader
    from dataloader.morl.offline_loader_subtrajectories import OfflineLoader
    dataset_path = config['xmorl']['dataset_path']
    loader = OfflineLoader(dataset_path=dataset_path,
                           cluster_id=cluster_id,
                            act_low = np.array(env.action_space.low),
                            act_high = np.array(env.action_space.high),
                            **config)



    optimizer = Optimizer(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps+1)/config['warmup_steps'], 1)
    )


    pref_loss = config['pref_loss']
    return_loss = config['return_loss']
    if (not pref_loss) and (not return_loss):
        loss_fn = lambda s_hat, a_hat, r_hat, pref_hat, s, a, r, pref: \
            torch.mean((a_hat - a) ** 2)
    # alternatively, can train on predicting preference
    elif (not pref_loss) and return_loss:
        loss_fn = lambda s_hat, a_hat, r_hat, pref_hat, s, a, r, pref: \
            torch.mean((a_hat - a) ** 2) + torch.mean((r_hat - r) ** 2)
    elif pref_loss and (not return_loss):
        loss_fn = lambda s_hat, a_hat, r_hat, pref_hat, s, a, r, pref: \
            torch.mean((a_hat - a) ** 2) + torch.mean((pref_hat - pref) ** 2)
    else:
        loss_fn = lambda s_hat, a_hat, r_hat, pref_hat, s, a, r, pref: \
            torch.mean((a_hat - a) ** 2) + torch.mean((r_hat - r) ** 2) + torch.mean((pref_hat - pref) ** 2)
        

    # from dataloader.morl.utils import pref_grid
    # prefs = pref_grid(pref_dim, granularity=config['granularity'])

    prefs = np.array([config['xmorl']['pref']])

    from evaluators.evaluator import Evaluator
    evaluator = Evaluator(env=env,
        loader=loader,
        prefs=prefs,
        **config
    )
    

    from trainer.modt_trainer import Trainer
    model_trainer = Trainer(model=model, 
                            loader=loader, 
                            optimizer=optimizer, 
                            scheduler=scheduler,
                            loss_fn=loss_fn,
                            evaluator=evaluator,
                               **config)
    best_results = None
    if config["eval_only"]:
        eval_results = model_trainer.test(
            model_path=config["eval"]["model_path"]
        )
        print("\n---- Evaluation Results ----\n")
        from pprint import pprint
        pprint(eval_results)
        # plot_eval_pareto(eval_results, log_dir=config['dir'],env_name=env_name, dataset=config['dataset'], train_returns=loader.returns_mo)
        # plot_eval_pareto(eval_results, log_dir=config['dir'], env_name=env_name, dataset=config['dataset'], train_returns=None)
    else:
        total_steps = 0
        
        if 'pretrain' in config and config['pretrain']:
            total_steps = config['eval']['checkpoint']
            model_path = f'{log_dir}/checkpoint_{total_steps}.pth'
            print(f'Loading pretrained model at checkpoint {total_steps}...\nmodel_path={model_path}\n')
            model_trainer.load_model(model_path, optimizer=True)
        
        if not(os.path.exists(log_dir)):
            os.makedirs(log_dir)
        for i in range(0, config['max_iters']):
            loss, eval_results = model_trainer.train_iteration(i)
            total_steps += config['num_steps_per_iter']
            model_path = f'{log_dir}/checkpoint_{total_steps}.pth'
            model_trainer.save_model(model_path)
            # plot_eval_pareto(eval_results, log_dir, iter=total_steps, env_name=env_name, dataset=config['dataset'], train_returns=loader.returns_mo)
            # with open(f'{log_dir}/eval_logs_{total_steps}.pth', 'w') as fp:
            #     json.dump(eval_results, fp)
            eval_results = eval_results[0]
            eval_results['checkpoint'] = total_steps
            if best_results is None:
                best_results = eval_results
            else:
                if eval_results['weighted returns'] > best_results['weighted returns']:
                    best_results = eval_results

        
    print(f"\n******* BEST Results - pref {config['xmorl']['pref']} - cluster {config['xmorl']['cluster_id']} *********\n", best_results, "\n****************")
    return best_results



if __name__ == '__main__':

    
    config_file = "configs/halfcheetah_v2/expert_uniform_pref.json"

    with open(config_file, "r") as f:
        config = json.load(f)

    pref_to_tag = {
        (0.0, 1.0): '000_100',
        (0.25, 0.75): '025_075',
        (0.5, 0.5): '050_050',
        (0.75, 0.25): '075_025',
        (1.0, 0.0): '100_000',
    }
    prefs = [list(key) for key in pref_to_tag.keys()]
    tags = list(pref_to_tag.values())
    
    cluster_file = os.path.join(config['dir'], "xmorl", "clusters.pkl")
    with open(cluster_file, 'rb') as f:
        cluster_info = pickle.load(f)


    comp_results = []
    config['xmorl'] = {}
    log_dir = config['dir']
    for cluster in cluster_info:
        pref_results = {}
        pref = cluster['pref']
        tag = pref_to_tag.get(tuple(pref), None)
        clusters = cluster['clusters']
        traj_cluster_labels = cluster['traj_cluster_labels']
        cluster_centers = cluster['cluster_centers']
        num_clusters = len(clusters)
        print(f"Num Clusters: {num_clusters}")
        # print(f"Pref: {pref}, Tag: {tag}, Num Clusters: {len(clusters)}")
        # for i, c in enumerate(clusters):
        #     # print(f"Cluster {i}: Size {len(c)}, Center: {cluster_centers[i]}")

        dataset_path = os.path.join(log_dir, "xmorl", tag, "subtrajs_eps_features.pkl")
        config['xmorl']['pref'] = pref
        config["num_steps_per_iter"] = 10000
        config["max_iters"] = 20
        cluster_ids = [None] + list(range(0, num_clusters))
        for cluster_id in cluster_ids:
            config['xmorl']['cluster_id'] = cluster_id
            config['xmorl']['dataset_path'] = dataset_path
            config['xmorl']['pref'] = pref
            cpolicy_eval = run(config=config)
            pref_results[f'cluster_{cluster_id}'] = cpolicy_eval
        comp_results.append({
            'pref': pref,
            'tag': tag,
            'results': pref_results
        })

    with open(os.path.join(config['dir'], "xmorl", "comp_policy_results.pkl"), 'wb') as f:
        pickle.dump(comp_results, f)