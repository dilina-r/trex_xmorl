import os
import numpy as np
import pandas as pd
import pickle
from math import isclose
import torch
import random

from .env_params import state_norm_params
from .utils import isCloseToOne, pref_grid

class OfflineLoader(object):
    def __init__(self,
            dataset_path,
            cluster_id,
            env_name,
            state_dim,
            act_dim,
            pref_dim,
            rtg_dim,
            model_type,
            batch_size,
            K,
            percent_dt,
            concat_state_pref,
            scale,
            concat_rtg_pref,
            concat_act_pref,
            normalize_reward,
            act_high,
            act_low,
            device,
            max_ep_len,
            max_length,
            gamma=0.1,
            avg_rtg = False,
            use_obj = -1,
            **kwargs):
        

        # self.dataset_path = f"{data_dir}/{env_name}/{env_name}_50000_{dataset}.pkl"


        self.percent_dt = percent_dt
        self.pref_dim = pref_dim
        self.state_dim = state_dim
        self.act_dim=act_dim
        self.rtg_dim = rtg_dim
        self.env_name=env_name
        self.max_length = max_length
        self.max_ep_len=max_ep_len
        self.batch_size=batch_size
        self.model_type = model_type
        self.device = device
        self.scale=scale
        self.normalize_reward = normalize_reward
        self.act_high=act_high
        self.act_low=act_low
        self.gamma=gamma
        self.use_obj=use_obj

        self.avg_rtg=bool(model_type == "rvs") 

        self.min_each_obj_step = 0
        self.max_each_obj_step = 1


        self.concat_state_pref=concat_state_pref
        self.concat_act_pref=concat_act_pref
        self.concat_rtg_pref=concat_rtg_pref


        self.trajectories = []
        with open(dataset_path, 'rb') as f:
            self.trajectories.extend(pickle.load(f))

        states, traj_lens, returns, returns_mo, preferences = [], [], [], [], []
        self.min_each_obj_step = np.min(np.vstack([np.min(traj['raw_rewards'], axis=0) for traj in self.trajectories]), axis=0)
        self.max_each_obj_step = np.max(np.vstack([np.max(traj['raw_rewards'], axis=0) for traj in self.trajectories]), axis=0)

        prev_state=None


        if cluster_id is not None:
            queries_trajectories = []
            for traj in self.trajectories:
                # print(traj.keys()); exit()
                if traj['cluster_id'] != cluster_id:
                    queries_trajectories.append(traj)

            print(f'\n\n\n\n Queried {len(queries_trajectories)} / {len(self.trajectories)} trajectories \n\n\n\n')
            self.trajectories = queries_trajectories



        for traj in self.trajectories:

            if concat_state_pref != 0:
                traj['observations'] = np.concatenate((traj['observations'], np.tile(traj['preference'], concat_state_pref)), axis=1)
                
            if normalize_reward:
                traj['raw_rewards'] = (traj['raw_rewards'] - self.min_each_obj_step) / (self.max_each_obj_step - self.min_each_obj_step)

            if self.rtg_dim==1:
                traj['rtgs'] = self.discount_cumsum(traj['rewards'])
            else:
                traj['rtgs'] = self.discount_cumsum_mo(traj['raw_rewards'])
            
            traj['rewards'] = np.sum(np.multiply(traj['raw_rewards'], traj['preference']), axis=1)
            states.append(traj['observations'])
            traj_lens.append(len(traj['observations']))
            returns.append(traj['rewards'].sum())
            returns_mo.append(traj['raw_rewards'].sum(axis=0))
            preferences.append(traj['preference'][0, :])



            # if traj['observations'].shape != prev_state:
            #     print(traj['observations'].shape)
            #     prev_state = traj['observations'].shape
        
        print(len(states))
        traj_lens = np.array(traj_lens)
        returns = np.array(returns)
        returns_mo = np.array(returns_mo)
        # states = np.array(states)
        preferences = np.array(preferences)

        # traj_lens, returns, returns_mo, states, preferences = np.array(traj_lens), np.array(returns), np.array(returns_mo), np.array(states), np.array(preferences)

        self.max_each_obj_traj=np.max(returns_mo, axis=0)
        # print(f"Teajectory lens: {traj['observations'].shape}"); exit()

        if not isCloseToOne(percent_dt):
            num_traj_wanted = int(percent_dt * len(self.trajectories))
            indices_wanted = np.unique(np.argpartition(returns_mo, -num_traj_wanted, axis=0)[-num_traj_wanted:])
            self.trajectories = np.array([self.trajectories[i] for i in indices_wanted])
            traj_lens = traj_lens[indices_wanted]
            returns = returns[indices_wanted]
            returns_mo = returns_mo[indices_wanted, :]
            states = states[indices_wanted]
            preferences = preferences[indices_wanted, :]
            

        states = np.concatenate(states, axis=0)
        self.state_mean = None
        self.state_std = None
        if env_name in state_norm_params:
            self.state_mean = state_norm_params[env_name]["mean"]
            self.state_std = np.sqrt(state_norm_params[env_name]["var"])
            self.state_mean = np.concatenate((self.state_mean, np.zeros(concat_state_pref * self.pref_dim)))
            self.state_std = np.concatenate((self.state_std, np.ones(concat_state_pref * self.pref_dim)))
        # self.state_dim += self.pref_dim * concat_state_pref

        self.lrModels = self.train_rtg_lr(preferences, returns_mo)
        
        max_prefs = np.max(preferences, axis=0)
        min_prefs = np.min(preferences, axis=0)
        # if concat_act_pref == 0 and concat_rtg_pref == 0 and concat_state_pref == 0 and model_type == "bc":
        #     granularity = 1
        # prefs = pref_grid(self.pref_dim, granularity=granularity); #print(f"Preferences: {prefs}")
        
        # print('=' * 50)
        # print(f'Starting new experiment: {env_name} {"_".join(dataset)}')
        # print(f'{len(traj_lens)} trajectories, {sum(traj_lens)} timesteps found')
        # print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        # print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        # print('=' * 50)

        self.returns_mo = returns_mo

        self.sorted_inds = np.argsort(returns)  # lowest to highest
        self.num_trajectories = len(traj_lens)
        self.p_sample = traj_lens[self.sorted_inds] / sum(traj_lens[self.sorted_inds])


    def discount_cumsum(self, x):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum

    def discount_cumsum_mo(self, x_mo):
        return np.transpose(np.array([self.discount_cumsum(x_mo[:,i]) for i in range(x_mo.shape[1])]))

    
    def find_avg_rtg(self, x):
        return np.mean(x)

    def find_avg_rtg_mo(self, x_mo):
        return np.mean(x_mo, axis=0)

    def __call__(self):
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=self.batch_size,
            replace=True,
            p=self.p_sample,
        )
        s, a, pref, rtg, timesteps, mask = [], [], [], [], [], []
        raw_r = []
        for i in batch_inds:
            # randomly get the traj from all trajectories
            traj = self.trajectories[int(self.sorted_inds[i])]
            # randomly get the starting idx
            step_start = random.randint(0, traj['rewards'].shape[0] - 1)
            step_end = step_start + self.max_length
            timestep_start = int(traj['timesteps']) + step_start

            s.append(traj['observations'][step_start:step_end].reshape(1, -1, self.state_dim))
            a.append(np.maximum(np.minimum(traj['actions'][step_start:step_end].reshape(1, -1, self.act_dim), self.act_high), self.act_low) / self.act_high) # assume scale if relflective to 0 (-x, x)
            raw_r_to_add = traj['raw_rewards'][step_start:step_end].reshape(1, -1, self.pref_dim)
            raw_r.append(raw_r_to_add)
            pref.append(traj['preference'][step_start:step_end].reshape(1, -1, self.pref_dim))
            timesteps.append(np.arange(timestep_start, timestep_start + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len-1  # padding cutoff
            
            # non-rvs: use discount cumsum

            # if 'rtgs' in traj:
            #     rtg.append(traj['rtgs'][step_start:step_end].reshape(1, -1, self.rtg_dim))
            # if rtg[-1].shape[1] <= s[-1].shape[1]:
            #     rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, self.rtg_dim))], axis=1)



            if self.rtg_dim == 1:
                rtg.append(self.discount_cumsum(traj['rewards'][step_start:step_end]).reshape(1, -1, self.rtg_dim))
            else:
                rtg.append(self.discount_cumsum_mo(traj['raw_rewards'][step_start:step_end]).reshape(1, -1, self.rtg_dim))
            
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, self.rtg_dim))], axis=1)



            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_length - tlen, self.state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate([np.ones((1, self.max_length - tlen, self.act_dim)) * -0., a[-1]], axis=1)
            raw_r[-1] = np.concatenate([np.zeros((1, self.max_length - tlen, self.pref_dim)), raw_r[-1]], axis=1)
            pref[-1] = np.concatenate([np.zeros((1, self.max_length - tlen, self.pref_dim)), pref[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_length - tlen, self.rtg_dim)), rtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_length - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_length - tlen)), np.ones((1, tlen))], axis=1))

            # print("dataloader -- ", tlen, rtg[-1].shape)

        if self.state_mean is not None:
            s = np.clip((s - self.state_mean) / self.state_std, -10, 10)
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        raw_r = torch.from_numpy(np.concatenate(raw_r, axis=0)).to(dtype=torch.float32, device=self.device) / self.scale
        pref = torch.from_numpy(np.concatenate(pref, axis=0)).to(dtype=torch.float32, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device) / self.scale
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)
        # print("dataloader -- ", rtg.shape)
        return s, a, raw_r, rtg, timesteps, mask, pref
    
    def train_rtg_lr(self, preferences, returns_mo):
        from sklearn.linear_model import LinearRegression
        # lrModels = [LinearRegression() for _ in range(self.pref_dim)]
        # for obj, lrModel in enumerate(lrModels):
        #     lrModel.fit(preferences.reshape((-1, self.pref_dim)), returns_mo[:, obj])

        
        # # all experiments use pre-cashed expert_uniform models
        with open(f"lr_models/{self.env_name}_expert_uniform.pkl", 'rb') as f:
            lrModels = pickle.load(f)

        return lrModels