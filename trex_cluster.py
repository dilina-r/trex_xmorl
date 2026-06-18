#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
default_n_threads = 64
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import argparse
import json
import pickle
import shutil

import gym
import numpy as np
import pandas as pd
import torch
import logging
import warnings

import seaborn as sns
from matplotlib import pyplot as plt
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn import preprocessing

import environments
from xmorl.episode_collector import EpisodeCollector

np.seterr(all="ignore")
np.warnings = warnings


def kmean_num_clusters(traj_embeddings, amount_initial_centers, max_clusters):
    range_n_clusters = list(range(amount_initial_centers, max_clusters + 1))
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
        cluster_labels = clusterer.fit_predict(traj_embeddings)
        silhouette_avg = silhouette_score(traj_embeddings, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        silhouette_scores.append(silhouette_avg)

    if len(silhouette_scores) == 0:
        return amount_initial_centers

    k = range_n_clusters[int(np.argmax(silhouette_scores))]
    return k


def perform_clustering_and_plot(traj_embeddings, amount_initial_centers, max_clusters, mode='xm', tag="", ccore=False, plot=True, log_dir=None):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info('Starting clustering process.')

    cluster_centers = None

    if mode == 'xm':
        print("Clustering using X-means")
        initial_centers = kmeans_plusplus_initializer(traj_embeddings, amount_initial_centers).initialize()
        logger.info('Initial centers initialized.')
        xmeans_instance = xmeans(traj_embeddings, initial_centers, max_clusters, ccore=ccore, criterion=1)
        xmeans_instance.process()
        logger.info('X-Means instance processed.')
        clusters = xmeans_instance.get_clusters()
    elif mode == 'kmeans':
        k = kmean_num_clusters(traj_embeddings, amount_initial_centers, max_clusters)
        print(f"\n\nKmeans Best K: {k}\n\n")
        clusterer = KMeans(n_clusters=k, random_state=10, n_init=10)
        cluster_labels = clusterer.fit_predict(traj_embeddings)
        clusters_dict = {}
        for i, label in enumerate(cluster_labels):
            clusters_dict.setdefault(label, []).append(i)
        cluster_centers = clusterer.cluster_centers_
        clusters = list(clusters_dict.values())
    else:
        print("Clustering using DBScan")
        eps = 0.5
        dbscan = DBSCAN(eps=eps, min_samples=amount_initial_centers).fit(traj_embeddings)
        clustering_labels = dbscan.labels_
        clusters_dict = {}
        for i, label in enumerate(clustering_labels):
            clusters_dict.setdefault(label, []).append(i)
        clusters = list(clusters_dict.values())
        cluster_centers = None

    logger.info('Clustering results extracted.')
    traj_cluster_labels = np.zeros(len(traj_embeddings), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for traj_id in cluster:
            traj_cluster_labels[traj_id] = cluster_id
    logger.info('Cluster labels assigned to each trajectory.')

    pca_traj = PCA(n_components=2)
    pca_traj_embeds = pca_traj.fit_transform(traj_embeddings)
    df = pd.DataFrame({
        'feature 1': pca_traj_embeds[:, 0],
        'feature 2': pca_traj_embeds[:, 1],
        'cluster id': traj_cluster_labels
    })
    logger.info('PCA performed for visualization.')

    if plot:
        plt.figure(figsize=(4, 3))
        palette = sns.color_palette('husl', len(clusters) + 1)
        sns.scatterplot(x='feature 1', y='feature 2', hue='cluster id', palette=palette[:len(clusters)], data=df, legend=True)
        plt.title('Trajectory Embeddings for ' + str(amount_initial_centers) + ' initial centers')
        plt.legend(title='$c_{j}$', loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=5)
        plt.tight_layout()
        if log_dir is not None:
            plt.savefig(os.path.join(log_dir, f'clusters_{tag}_n{len(clusters)}.png'))
        else:
            plt.savefig(f'clusters_{tag}.png')
        logger.info('Plot created.')

    return clusters, traj_cluster_labels, cluster_centers


def discount_cumsum(x, gamma):
    result = np.zeros_like(x)
    result[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        result[t] = x[t] + gamma * result[t + 1]
    return result


def discount_cumsum_mo(x_mo, gamma):
    return np.transpose(np.array([discount_cumsum(x_mo[:, i], gamma) for i in range(x_mo.shape[1])]))


def get_trajectory_embedding(model, config, obs, actions, raw_returns, timesteps, prefs, state_mean, state_std, gamma=1.0, embedding_mode="concat"):
    device = config['device']
    state_dim = config['state_dim']
    act_dim = config['act_dim']
    rtg_dim = config['rtg_dim']

    model.eval()
    with torch.no_grad():
        state_np = np.concatenate((obs, np.tile(prefs, config['concat_state_pref'])), axis=1)
        state_np = np.clip((state_np - state_mean) / state_std, -10, 10)
        input_obs = torch.from_numpy(state_np.reshape(1, -1, state_dim)).float()
        input_act = torch.from_numpy(np.array(actions).reshape(1, -1, act_dim)).float()
        input_rew = torch.from_numpy(discount_cumsum_mo(np.array(raw_returns), gamma=gamma).reshape(1, -1, rtg_dim)).float()
        timestamps = torch.from_numpy(np.array(timesteps).reshape(1, -1, 1)).long()

        embeddings = model.get_embeddings(states=input_obs.to(device),
                                        actions=input_act.to(device),
                                        returns_to_go=input_rew.to(device),
                                        pref=None,
                                        timesteps=timestamps.to(device))

        if embedding_mode == "concat":
            embeddings = torch.mean(embeddings, dim=1).flatten()
        else:
            embeddings = torch.mean(embeddings.reshape(-1, embeddings.shape[-1]), dim=0)

    return embeddings.cpu()


def create_subtrajectories(traj_file, l=30, margin=5):
    with open(traj_file, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"Loaded {len(trajectories)} trajectories from {traj_file}")
    subtrajectories = []

    for traj in trajectories:
        n = len(traj['observations'])
        print(f"Trajectory {traj.get('episode_id', 'unknown')} length = {n}")
        inc = l - margin
        i = 0
        while (i + l) <= n:
            subtraj = {
                'episode_id': traj['episode_id'],
                'observations': traj['observations'][i:i + l],
                'actions': traj['actions'][i:i + l],
                'raw_rewards': traj['raw_rewards'][i:i + l],
                'timesteps': i,
                'preference': traj['preference'][i:i + l]
            }
            subtrajectories.append(subtraj)
            i += inc

    print(f"Created {len(subtrajectories)} subtrajectories")
    return subtrajectories


def sample_and_save_features(config, target_pref, tag, epsilon, num_episodes, l=20, margin=5):
    dataset_path = f"{config['data']}/{config['env_name']}/{config['env_name']}_50000_{config['dataset']}.pkl"
    model_path = config['xrl']['pretrained_model']
    env = gym.make(config['env_name'])

    device = config['device']
    act_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    reward_size = env.obj_dim
    pref_dim = reward_size
    state_dim += pref_dim * config['concat_state_pref']
    rtg_dim = pref_dim

    config['state_dim'] = state_dim
    config['act_dim'] = act_dim
    config['pref_dim'] = pref_dim
    config['rtg_dim'] = rtg_dim

    print(f"Sampling with target preference {target_pref} and tag {tag}")
    n_inner = 4 * config['hidden_size']
    log_dir = config['dir']
    os.makedirs(log_dir, exist_ok=True)

    from models.modt.decision_transformer import DecisionTransformer
    model = DecisionTransformer(n_inner=n_inner, n_positions=1024, **config).to(device)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict['model'])

    collector = EpisodeCollector(env=env, **config)

    for i in range(num_episodes):
        print(f"Running Episode : {i+1} / {num_episodes}")
        collector.run_episode(model, target_pref, episode_len=500, epsilon=epsilon, record_frame=True)

    out_dir = os.path.join(log_dir, 'xmorl', tag)
    os.makedirs(out_dir, exist_ok=True)
    buffer_filepath = os.path.join(out_dir, f'buffer_{num_episodes}eps.pkl')
    collector.save_replay_buffer(filepath=buffer_filepath)
    print(f"Saved buffer to {buffer_filepath}")

    subtrajectories = create_subtrajectories(traj_file=buffer_filepath, l=l, margin=margin)
    traj_path = os.path.join(out_dir, f'subtrajs_eps.pkl')
    with open(traj_path, 'wb') as f:
        pickle.dump(subtrajectories, f)

    for i, subtraj in enumerate(subtrajectories):
        emb = get_trajectory_embedding(model,
                                       config,
                                       obs=subtraj['observations'],
                                       actions=subtraj['actions'],
                                       raw_returns=subtraj['raw_rewards'],
                                       timesteps=list(range(subtraj['timesteps'], subtraj['timesteps'] + l)),
                                       prefs=subtraj['preference'],
                                       state_mean=collector.state_mean,
                                       state_std=collector.state_std,
                                       gamma=1.0,
                                       embedding_mode=config['xrl'].get('embedding_mode', 'mean'))
        subtraj['feature_embd'] = emb.numpy() if isinstance(emb, torch.Tensor) else emb

    feature_path = os.path.join(out_dir, f'subtrajs_eps_features.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(subtrajectories, f)
    print(f"Saved feature-enhanced subtrajectories to {feature_path}")

    return feature_path


def cluster_and_analyze(config, prefs, tags, amount_initial_centers=3, max_clusters=10):
    log_dir = config['dir']
    plot_dir = os.path.join(log_dir, 'plots')
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    pref_cluster_centers = []
    cluster_info = []
    all_embeddings = []

    for tag in tags:
        filename = os.path.join(config['dir'], 'xmorl', tag, 'subtrajs_eps_features.pkl')
        with open(filename, 'rb') as f:
            subtrajectories = pickle.load(f)
        for subtraj in subtrajectories:
            all_embeddings.append(subtraj['feature_embd'])

    all_embeddings = np.array(all_embeddings)
    print(f"Loaded {len(all_embeddings)} total embeddings")
    if all_embeddings.size > 0:
        scaler = preprocessing.MinMaxScaler()
        X_train_minmax = scaler.fit_transform(all_embeddings)
        print(f"Scaled embeddings shape: {X_train_minmax.shape}")

    for p, (pref, tag) in enumerate(zip(prefs, tags)):
        filename = os.path.join(config['dir'], 'xmorl', tag, 'subtrajs_eps_features.pkl')
        with open(filename, 'rb') as f:
            subtrajectories = pickle.load(f)

        embeddings = np.array([np.array(subtraj['feature_embd']) for subtraj in subtrajectories])
        clusters, traj_cluster_labels, cluster_centers = perform_clustering_and_plot(
            traj_embeddings=embeddings,
            amount_initial_centers=amount_initial_centers,
            max_clusters=max_clusters,
            mode='kmeans',
            tag=tag,
            plot=True,
            log_dir=plot_dir
        )

        for s, subtraj in enumerate(subtrajectories):
            subtraj['cluster_id'] = int(traj_cluster_labels[s])

        with open(filename, 'wb') as f:
            pickle.dump(subtrajectories, f)

        print(f"Saved clustered subtrajectories to {filename}")
        pref_cluster_centers.append(cluster_centers)
        cluster_info.append({
            'pref': pref,
            'clusters': clusters,
            'traj_cluster_labels': traj_cluster_labels,
            'cluster_centers': cluster_centers
        })

    cluster_file = os.path.join(config['dir'], 'xmorl', 'clusters.pkl')
    with open(cluster_file, 'wb') as f:
        pickle.dump(cluster_info, f)
    print(f"Saved cluster metadata to {cluster_file}")
    return cluster_file


def main():
    # parser = argparse.ArgumentParser(description='Run end-to-end TREX sampling and clustering analysis.')
    # parser.add_argument('--config', type=str, default='configs/ant_v2/expert_uniform_pref_paper.json', help='Path to the JSON config file')
    # parser.add_argument('--num_episodes', type=int, default=25, help='Number of episodes to sample per preference')
    # parser.add_argument('--epsilon', type=float, default=0.05, help='Exploration epsilon during episode collection')
    # parser.add_argument('--prefs', nargs='+', default=None, help='Optional list of preference vectors, e.g. 0.0,1.0 0.25,0.75')
    # parser.add_argument('--tags', nargs='+', default=None, help='Optional list of tags for each preference')
    # parser.add_argument('--max_clusters', type=int, default=10, help='Maximum number of clusters to consider')
    # args = parser.parse_args()
    config_file = "/home/dilina/XAI/TREX/configs/halfcheetah_v2/expert_uniform_pref.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    prefs = [[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]]
    tags = ['000_100', '025_075', '050_050', '075_025', '100_000']
    epsilon = 0.05
    num_episodes = 25
    max_clusters = 10

    for pref, tag in zip(prefs, tags):
        target_pref = np.array(pref)
        sample_and_save_features(config=config, target_pref=target_pref, tag=tag, epsilon=epsilon, num_episodes=num_episodes)

    cluster_and_analyze(config=config, prefs=prefs, tags=tags, max_clusters=max_clusters)


if __name__ == '__main__':
    main()
