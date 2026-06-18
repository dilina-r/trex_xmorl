import numpy as np
import torch
import time
from tqdm import tqdm
class Trainer:

    def __init__(
        self,
        model,
        loader,
        optimizer,
        loss_fn,
        evaluator,
        scheduler=None,
        max_iter=0,
        num_steps_per_iter=0,
        eval_only=False,
        concat_rtg_pref=0,
        concat_act_pref=0,
        **kwargs
    ):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # for plotting purposes
        self.scheduler = scheduler
        self.evaluator=evaluator
        self.max_iter = max_iter
        self.num_steps_per_iter = num_steps_per_iter
        self.eval_only = eval_only
        self.concat_rtg_pref = concat_rtg_pref
        self.concat_act_pref = concat_act_pref
        # self.logsdir = logsdir
        self.diagnostics = dict()
        self.start_time = time.time()
        self.best_results = None

    def train_iteration(self, ep):
        train_losses = []
        logs = dict()
        train_start = time.time()
        if not self.eval_only:
            self.model.train()
            for ite in tqdm(range(self.num_steps_per_iter), disable=True):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
                    

        logs['time/training'] = time.time() - train_start
        eval_start = time.time()
        self.model.eval()
        cur_step = (ep+1) * self.num_steps_per_iter



        # rollout_unweighted_raw_r = np.array(set_unweighted_raw_return)
        # rollout_weighted_raw_r = np.array(set_weighted_raw_return)
        # rollout_original_raw_r = np.array(set_cum_r_original)
        # target_prefs = np.array([eval_fn.target_pref for eval_fn in self.eval_fns])
        # target_returns = np.array([eval_fn.target_reward for eval_fn in self.eval_fns]) # target returns are weighted

        
        
        # n_obj = self.model.pref_dim
        # # rollout_ratio = rollout_original_raw_r / np.sum(rollout_original_raw_r, axis=1, keepdims=True)
        # # rollout_logs = {
        # #     'n_obj': n_obj,
        # #     'target_prefs': target_prefs,
        # #     'target_returns': target_returns,
        # #     'dataset_min_prefs': self.dataset_min_prefs,
        # #     'dataset_max_prefs': self.dataset_max_prefs,
        # #     'dataset_min_raw_r': self.dataset_min_raw_r,
        # #     'dataset_max_raw_r': self.dataset_max_raw_r,
        # #     'dataset_min_final_r': self.dataset_min_final_r,
        # #     'dataset_max_final_r': self.dataset_max_final_r,
        # #     'rollout_unweighted_raw_r': rollout_unweighted_raw_r,
        # #     'rollout_weighted_raw_r': rollout_weighted_raw_r, # for finding [achieved return vs. target return]
        # #     'rollout_original_raw_r': rollout_original_raw_r, # unnormalized raw_r, for calculating roll-out ratio
        # # }
        
        # # visualize(rollout_logs, self.logsdir, cur_step)
        
        
        print(f"\n------------------> epoch: {ep} <------------------")
        print(f"loss = {np.mean(train_losses)}")

        eval_results = self.evaluator.evaluate(self.model)
        print(eval_results, "\n")

        return np.mean(train_losses), eval_results


    def train_step(self):
        states, actions, raw_return, rtg, timesteps, attention_mask, pref = self.loader()
        rtg = rtg[:, :-1]
        
        action_target = torch.clone(actions)
        return_target = torch.clone(raw_return)
        pref_target = torch.clone(pref)

        # print("modt_trainer -- ", rtg.shape)

        
        if self.concat_rtg_pref != 0:
            rtg = torch.cat((rtg, torch.cat([pref] * self.concat_rtg_pref, dim=2)), dim=2)
        if self.concat_act_pref != 0:
            actions = torch.cat((actions, torch.cat([pref] * self.concat_act_pref, dim=2)), dim=2)

        
        action_preds, return_preds, pref_preds = self.model.forward(
            states, actions, rtg, pref, timesteps, attention_mask=attention_mask
        )

        act_dim = self.loader.act_dim
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        pref_dim = self.loader.pref_dim
        return_preds = return_preds.reshape(-1, pref_dim)[attention_mask.reshape(-1) > 0]
        return_target = return_target.reshape(-1, pref_dim)[attention_mask.reshape(-1) > 0]
        
        pref_preds = pref_preds.reshape(-1, pref_dim)[attention_mask.reshape(-1) > 0]
        pref_target = pref_target.reshape(-1, pref_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, return_preds, pref_preds,
            None, action_target, return_target, pref_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            self.diagnostics['training/return_error'] = torch.mean((return_preds - return_target) ** 2).detach().cpu().item()
            self.diagnostics['training/pref_error'] = torch.mean((pref_preds - pref_target) ** 2).detach().cpu().item()
            
        return loss.detach().cpu().item()
    

    def test(self, model_path):
        self.load_model(model_path)
        eval_results = self.evaluator.evaluate(self.model)
        return eval_results

    
    def save_model(self, filepath):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)

    def load_model(self, filepath, optimizer=False):
        state_dict = torch.load(filepath, weights_only=True)
        self.model.load_state_dict(state_dict['model'])

        if optimizer and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])