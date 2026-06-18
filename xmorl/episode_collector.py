import os
from copy import deepcopy
import numpy as np
import torch
from collections import defaultdict
import gym
from gymnasium.wrappers import RecordVideo
import pickle
import uuid

import imageio
from .env_params import state_norm_params
from .replay_buffer import LoggingReplayBuffer

class EpisodeCollector:

    def __init__(self, 
                env, 
                env_name,
                state_dim,
                act_dim,
                rtg_dim,
                pref_dim,
                max_ep_len,
                concat_state_pref,
                concat_act_pref,
                concat_rtg_pref,
                device,
                use_max_rtg,
                dir,
                seed=None,
                scale=1.0,
                **kwargs):
        self.eval_env=env
        self.env_name=env_name
        self.eval_env.reset()
        self.state_dim=state_dim
        self.act_dim=act_dim
        self.pref_dim=pref_dim
        self.rtg_dim=rtg_dim
        self.max_ep_len=max_ep_len
        self.scale=scale
        self.concat_state_pref=concat_state_pref
        self.concat_rtg_pref=concat_rtg_pref
        self.concat_act_pref=concat_act_pref
        self.device=device
        self.use_max_rtg=use_max_rtg
        self.log_dir=dir
        self.seed = seed


        self.state_mean = state_norm_params[self.env_name]["mean"]
        self.state_std = np.sqrt(state_norm_params[self.env_name]["var"])
        self.state_mean = np.concatenate((self.state_mean, np.zeros(self.concat_state_pref * self.pref_dim)))
        self.state_std = np.concatenate((self.state_std, np.ones(self.concat_state_pref * self.pref_dim)))


        ## Load LR models to determine RTG 
        with open(f"lr_models/{self.env_name}_expert_uniform.pkl", 'rb') as f:
            self.lrModels = pickle.load(f)

        self.buffer = LoggingReplayBuffer()

    
        
    def run_episode(self, model, target_pref, episode_len, epsilon=0, record_frame=False):

        model.eval()
        model.to(device=self.device)

        ### Get expected RTG from pref

        n_obj = self.rtg_dim
        target_return = np.array([self.lrModels[i].predict(target_pref.reshape(-1, n_obj))[0] for i in range(n_obj)])
        target_return = np.array(target_return)
        target_return = np.multiply(target_return, target_pref)

        if self.seed is None:
            self.seed = np.random.randint(0, 10000)

        episode_id = uuid.uuid1()
        self.buffer.reset_episode(episode_id=episode_id, seed=self.seed)

        with torch.no_grad():
            init_target_return = deepcopy(target_return)

            init_target_pref = deepcopy(target_pref)
            
            state_mean = torch.from_numpy(self.state_mean).to(device=self.device, dtype=torch.float32)
            state_std = torch.from_numpy(self.state_std).to(device=self.device, dtype=torch.float32)
            
            
            self.eval_env.seed(self.seed) # fixed seeding in evaluation to visualize
            state_np = self.eval_env.reset()
            curr_state  = state_np

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
            frames = []
            for t in range(episode_len):
                # add padding
                actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)

                action = model.get_action(
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    prefs.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )

                try:
                    frame = self.eval_env.render(mode='rgb_array')
                    frames.append(frame)
                except Exception as e:
                    print(f"Rendering failed at step {t}: {e}")

                actions[-1] = action
                action = action.detach().cpu().numpy()
                # action = np.multiply(action, self.act_scale)

                if epsilon > 0:
                    random_act = np.random.choice([True, False], size=1, replace=False, p=[epsilon, 1-epsilon])[0]
                    if random_act:
                        action = self.eval_env.action_space.sample()

                state_np, _, done, info = self.eval_env.step(action)

                self.buffer.add(curr_state, state_np, action, target_pref, t, info['obj'], done)
                curr_state=state_np
                
                # eval: for return, don't process any data, NO clipping, NO rewriting, etc.
                # model: for auto-reg rollout, process data
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

            if len(frames) > 0 and record_frame:
                os.makedirs(os.path.join(self.log_dir,'xmorl', 'render_videos'), exist_ok=True)
                # frames_path = os.path.join(os.path.join(self.log_dir, 'render_videos', f'{episode_id}.npy'))
                # frames = np.array(frames)
                # np.save(frames_path, frames)
            
                
                output_filename = os.path.join(os.path.join(self.log_dir, 'xmorl', 'render_videos', f'{episode_id}.mp4'))
                fps = 30
                print(f"Saving video to {output_filename} ({len(frames)} frames)...")
                imageio.mimsave(output_filename, frames, fps=fps)

            target_ret_scaled_back = np.round(init_target_return * self.scale, 3) # this is normalized
            weighted_raw_reward_cumulative_eval = np.round(np.multiply(unweighted_raw_reward_cumulative_eval * self.scale, init_target_pref), 3)
            unweighted_raw_return_cumulative_eval = np.round(unweighted_raw_reward_cumulative_eval * self.scale, 3)
            # print(unweighted_raw_return_cumulative_eval); exit()
            total_return_scaled_back_eval = np.round(np.sum(weighted_raw_reward_cumulative_eval), 3)
            # if not self.eval_only:
            # if not self.eval_only:
            #     log_file_name = os.path.join(self.log_dir,f'step={cur_step}.txt')
            #     with open(log_file_name, 'a') as f:
            #         f.write(f"\ntarget return: {target_ret_scaled_back} ------------> {weighted_raw_reward_cumulative_eval}\n")
            #         f.write(f"target pref: {np.round(init_target_pref, 3)} ------------> {np.round(cum_r_original / np.sum(cum_r_original), 3)}\n")
            #         f.write(f"\tunweighted raw returns: {unweighted_raw_return_cumulative_eval}\n")
            #         f.write(f"\tweighted raw return: {weighted_raw_reward_cumulative_eval}\n")
            #         f.write(f"\tweighted final return: {total_return_scaled_back_eval}\n")
            #         f.write(f"\tlength: {episode_length_eval}\n")
            
            # env.close()
            # self.decide_save_video(np.multiply(actions.detach().cpu().numpy(), self.act_scale), raw_rewards_cumulative, init_target_return, init_target_pref, seed)
            return episode_return_eval, episode_length_eval, unweighted_raw_return_cumulative_eval, weighted_raw_reward_cumulative_eval, cum_r_original
        

    def save_replay_buffer(self, filepath):
        self.buffer.save(filename=filepath)

        