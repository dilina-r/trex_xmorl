import os
from copy import deepcopy
import numpy as np
import torch
from collections import defaultdict
import gym
from gymnasium.wrappers import RecordVideo

class Evaluator:
    
    def __init__(
        self,
        env,
        loader,
        prefs,
        use_obj,
        device,
        mode,
        eval_only=False, 
        act_scale=None, 
        num_eval_episodes=1,
        use_max_rtg=True,
        rtg_scale=1.0,
        max_ep_len=500,
        dir="",
        seed=1,
        **kwargs
    ):
        self.eval_env=env
        self.eval_env.reset()
        self.loader = loader
        self.prefs = prefs
        self.state_dim=self.loader.state_dim
        self.act_dim=self.loader.act_dim
        self.pref_dim=self.loader.pref_dim
        self.rtg_dim=self.loader.rtg_dim
        self.max_ep_len=max_ep_len
        self.scale=self.loader.scale
        self.state_mean=self.loader.state_mean
        self.state_std=self.loader.state_std
        self.min_each_obj_step=self.loader.min_each_obj_step
        self.max_each_obj_step=self.loader.max_each_obj_step
        self.max_each_obj_traj = self.loader.max_each_obj_traj
        self.act_scale=act_scale if act_scale is not None else np.ones(shape=self.act_dim)
        self.use_obj=use_obj
        self.concat_state_pref=self.loader.concat_state_pref
        self.concat_rtg_pref=self.loader.concat_rtg_pref
        self.concat_act_pref=self.loader.concat_act_pref
        self.normalize_reward=self.loader.normalize_reward
        self.device=device
        self.mode=mode
        self.eval_only=eval_only
        self.use_max_rtg=use_max_rtg
        self.rtg_scale=rtg_scale
        self.log_dir=dir
        self.seed = seed

        self.num_eval_episodes = num_eval_episodes

        
    def __eval_episode(self, model, target_return, target_pref, cur_step):

        # if self.eval_only:
        #     # Create the directory if it doesn't exist
        #     video_directory = os.path.join(self.log_dir, 'videos')
        #     os.makedirs(video_directory, exist_ok=True)
        #     env = RecordVideo(
        #             self.eval_env,
        #             video_folder=video_directory,
        #             name_prefix="eval-run",
        #             episode_trigger=lambda e: True  # <-- This change records EVERY episode
        #         )
        # else:
        #     env = deepcopy(self.eval_env)

        model.eval()
        model.to(device=self.device)
        # add a little variance to make sure eval results are not by luck
        # target_pref += np.random.normal(loc=0.0, scale=0.001, size=target_pref.shape)
        # target_pref = target_pref / np.sum(target_pref)
        # target_return += np.random.normal(loc=0.0, scale=0.001, size=target_return.shape)

        with torch.no_grad():
            init_target_return = deepcopy(target_return)

            init_target_pref = deepcopy(target_pref)
            
            state_mean = torch.from_numpy(self.state_mean).to(device=self.device, dtype=torch.float32)
            state_std = torch.from_numpy(self.state_std).to(device=self.device, dtype=torch.float32)
            
            if self.seed is None:
                self.seed = np.random.randint(0, 10000)
            self.eval_env.seed(self.seed) # fixed seeding in evaluation to visualize
            state_np = self.eval_env.reset()

            state_np = np.concatenate((state_np, np.tile(init_target_pref, self.concat_state_pref)), axis=0)
            
            state_tensor = torch.from_numpy(state_np).to(device=self.device, dtype=torch.float32).reshape(1, self.state_dim)
            state_tensor = torch.clip((state_tensor - state_mean) / state_std, -10, 10)
            states = state_tensor
            
            # if self.mode == 'noise':
            #     state = state + np.random.normal(0, 0.1, size=state.shape).astype(np.float32)
            
            actions = torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32)
            # prefs = torch.zeros((0, self.pref_dim), device=self.device, dtype=torch.float32)

            # prefs_to_go = torch.from_numpy(target_pref).to(device=self.device, dtype=torch.float32).reshape(1, self.pref_dim)
            pref_np = np.array(target_pref)
            pref_tensor = torch.from_numpy(pref_np).reshape(1, self.pref_dim).to(device=self.device, dtype=torch.float32)
            prefs = pref_tensor
            
            target_return = torch.tensor(target_return, device=self.device, dtype=torch.float32).reshape(1, self.rtg_dim)
            timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
            
            episode_return_eval, episode_length_eval = 0, 0
            unweighted_raw_reward_cumulative_eval = np.zeros(shape=(self.pref_dim), dtype=np.float32)
            unweighted_raw_reward_cumulative_model = np.zeros(shape=(self.pref_dim), dtype=np.float32)
            
            cum_r_original = np.zeros(shape=(self.pref_dim), dtype=np.float32)
            # print(f"Evaluation MAX EP LEN: {self.max_ep_len}")
            for t in range(self.max_ep_len):
                # add padding
                actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)

                action = model.get_action(
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    prefs.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
                actions[-1] = action
                action = action.detach().cpu().numpy()
                action = np.multiply(action, self.act_scale)

                state_np, _, done, info = self.eval_env.step(action)
                
                
                # eval: for return, don't process any data, NO clipping, NO rewriting, etc.
                # model: for auto-reg rollout, process data
                if self.normalize_reward:
                    unweighted_raw_reward_eval = (info['obj'] - self.min_each_obj_step) / (self.max_each_obj_step - self.min_each_obj_step) / self.scale
                    unweighted_raw_reward_model = np.clip((info['obj'] - self.min_each_obj_step) / (self.max_each_obj_step - self.min_each_obj_step), 0, 1) / self.scale
                else:
                    unweighted_raw_reward_eval = info['obj'] / self.scale
                    unweighted_raw_reward_model = info['obj'] / self.scale


                    
                cum_r_original += info['obj']
                
                final_reward_eval = np.dot(init_target_pref, unweighted_raw_reward_eval)
                final_reward_model = np.dot(init_target_pref, unweighted_raw_reward_model)
                weighted_raw_reward_eval = np.multiply(init_target_pref, unweighted_raw_reward_eval)
                weighted_raw_reward_model = np.multiply(init_target_pref, unweighted_raw_reward_model)
                unweighted_raw_reward_cumulative_eval += unweighted_raw_reward_eval
                unweighted_raw_reward_cumulative_model += unweighted_raw_reward_model
                
                state_np = np.concatenate((state_np, np.tile(init_target_pref, self.concat_state_pref)), axis=0)
                state_tensor = torch.from_numpy(state_np).to(device=self.device, dtype=torch.float32).reshape(1, self.state_dim)
                state_tensor = torch.clip((state_tensor - state_mean) / state_std, -10, 10)
                states = torch.cat([states, state_tensor], dim=0)
                prefs = torch.cat([prefs, pref_tensor], dim=0)

                

                unweighted_raw_reward_model = torch.from_numpy(np.array(unweighted_raw_reward_model)).to(device=self.device).reshape(1, self.pref_dim)
                weighted_raw_reward_model = torch.from_numpy(np.array(weighted_raw_reward_model)).to(device=self.device).reshape(1, self.pref_dim)

                
                if self.rtg_dim == 1:
                    pred_return = target_return[-1] - final_reward_model
                else:
                    pred_return = target_return[-1] - weighted_raw_reward_model
                target_return = torch.cat([target_return, pred_return.reshape(1, self.rtg_dim)], dim=0)
                timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (t+1)], dim=1)

                # MODT: find final reward through dot product
                episode_return_eval += final_reward_eval
                episode_length_eval += 1

                if done:
                    break

            target_ret_scaled_back = np.round(init_target_return * self.scale, 3) # this is normalized
            weighted_raw_reward_cumulative_eval = np.round(np.multiply(unweighted_raw_reward_cumulative_eval * self.scale, init_target_pref), 3)
            unweighted_raw_return_cumulative_eval = np.round(unweighted_raw_reward_cumulative_eval * self.scale, 3)
            # print(unweighted_raw_return_cumulative_eval); exit()
            total_return_scaled_back_eval = np.round(np.sum(weighted_raw_reward_cumulative_eval), 3)
            # if not self.eval_only:
            if not self.eval_only:
                log_file_name = os.path.join(self.log_dir,f'step={cur_step}.txt')
                with open(log_file_name, 'a') as f:
                    f.write(f"\ntarget return: {target_ret_scaled_back} ------------> {weighted_raw_reward_cumulative_eval}\n")
                    f.write(f"target pref: {np.round(init_target_pref, 3)} ------------> {np.round(cum_r_original / np.sum(cum_r_original), 3)}\n")
                    f.write(f"\tunweighted raw returns: {unweighted_raw_return_cumulative_eval}\n")
                    f.write(f"\tweighted raw return: {weighted_raw_reward_cumulative_eval}\n")
                    f.write(f"\tweighted final return: {total_return_scaled_back_eval}\n")
                    f.write(f"\tlength: {episode_length_eval}\n")
            
            # env.close()
            # self.decide_save_video(np.multiply(actions.detach().cpu().numpy(), self.act_scale), raw_rewards_cumulative, init_target_return, init_target_pref, seed)
            return episode_return_eval, episode_length_eval, unweighted_raw_return_cumulative_eval, weighted_raw_reward_cumulative_eval, cum_r_original
        

    def evaluate(self, model):

        if self.use_max_rtg:
            adjusted_target_rewards = np.multiply(self.max_each_obj_traj, self.prefs)
        else:
            adjusted_target_rewards = []
            n_obj = self.prefs.shape[1]
            for pref in self.prefs:
                adjusted_target_rewards.append(np.array([self.loader.lrModels[i].predict(pref.reshape(-1, n_obj))[0] for i in range(n_obj)]))
            adjusted_target_rewards = np.array(adjusted_target_rewards)
            adjusted_target_rewards = np.multiply(adjusted_target_rewards, self.prefs)

        # adjusted_target_rewards = np.array([[233.001, 73.352]])
        # adjusted_target_rewards = np.array([[51.324, 149.375]]) ## for w=(0.25, 0.75)
        results = {}
        
        returns = np.zeros(shape=(self.num_eval_episodes))
        lengths = np.zeros(shape=(self.num_eval_episodes))
        raw_returns = np.zeros(shape=(self.num_eval_episodes, self.pref_dim))
        weighted_raw_returns = np.zeros(shape=(self.num_eval_episodes, self.pref_dim))
        all_cum_r_original = np.zeros(shape=(self.num_eval_episodes, self.pref_dim))
        episode_count=0
        for i, target_pref in enumerate(self.prefs):
            target_reward=adjusted_target_rewards[i] * self.rtg_scale,
            target_pref=target_pref * self.rtg_scale
            for eval_ep in range(self.num_eval_episodes):
                ret, length, raw_ret, weighted_raw_ret, cum_r_original = self.__eval_episode(model, target_reward, target_pref, i)
                returns[eval_ep] = ret
                raw_returns[eval_ep, :] = raw_ret
                weighted_raw_returns[eval_ep, :] = weighted_raw_ret
                all_cum_r_original[eval_ep, :] = cum_r_original
                lengths[eval_ep] = length
            
        
                returns *= self.scale
                raw_returns *= self.scale
                # weighted_raw_returns *=self.scale

                target_reward = np.round(target_reward, decimals=0) # round to int each entry
                # raw_returns = np.round(raw_returns, decimals=1) # round to int each entry
                weighted_raw_returns = np.round(weighted_raw_returns, decimals=1) # round to int each entry
                # info for weighted return
                # target_ret_scaled_back = np.round(init_target_return * self.scale, 3) # this is normalized

                # infos = {
                #     f'total_return_mean/rtg_{target_reward}_pref_{target_pref}': np.mean(returns),
                #     f'length_mean/rtg_{target_reward}_pref_{target_pref}': np.mean(lengths),
                #     f'total_raw_return_mean/rtg_{target_reward}_pref_{target_pref}': np.mean(raw_returns),
                #     f'total_weighted_return_mean/rtg_{target_reward}_pref_{target_pref}': np.mean(weighted_raw_returns),
                # }
                # print(infos)

                # print(f'total_return_mean/rtg_{target_reward}_pref_{target_pref}: {list(returns)}')
                # print(f'length_mean/rtg_{target_reward}_pref_{target_pref} : {np.mean(lengths)}')
                # print(f'total_raw_return_mean/rtg_{target_reward}_pref_{target_pref} : {list(raw_returns[eval_ep, :])}')
                # print(f'total_weighted_return_mean/rtg_{target_reward}_pref_{target_pref} : {list(weighted_raw_returns[eval_ep, :])}')
                
                results[episode_count] = {
                    "preferences" : list(target_pref),
                    "target_reward" : list(np.squeeze(target_reward)),
                    # "returns" : np.mean(returns),
                    "raw_returns" : list(raw_ret),
                    # "lengths" : np.mean(lengths),
                    "weighted returns" : np.mean(weighted_raw_returns)
                    # "preference" : list(self.prefs[i])
                }
                episode_count+=1

        return results
    
