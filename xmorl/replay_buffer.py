import numpy as np
import os
import torch
import pickle

class LoggingReplayBuffer:
    """
    A Replay Buffer designed to log agent experiences during runtime
    and save them to disk for future Offline RL training.
    """
    def __init__(self):

        # # Pre-allocate memory for efficiency
        # self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        # self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        # self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        # self.reward = np.zeros((max_size, reward_dim), dtype=np.float32)
        # self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        self.states = []
        self.actions = []
        self.timestep = []
        self.rewards = []
        self.next_state = []
        self.done = []
        self.preference = []
        self.curr_buffer_size = 0
        self.current_id = None
        self.seed = None

        self.trajectories = []
        
    def reset_episode(self, episode_id, seed):
        if self.current_id is not None:
            traj = {
                'episode_id' : self.current_id,
                'observations' : np.array(self.states),
                'next_states' : np.array(self.next_state),
                'actions' : np.array(self.actions),
                'raw_rewards' : np.array(self.rewards),
                'timesteps' : self.timestep,
                'done' : self.done,
                'preference' : np.array(self.preference),
                'seed' : self.seed
            }
            self.trajectories.append(traj)
        self.current_id = episode_id
        self.seed = seed
        self.states = []
        self.actions = []
        self.timestep = []
        self.rewards = []
        self.next_state = []
        self.done = []
        self.preference = []

    def add(self, state, next_state, action, prefs, timestep, reward, done):
        """
        Add a new experience to the buffer.
        """
        # self.state[self.ptr] = state
        # self.action[self.ptr] = action
        # self.next_state[self.ptr] = next_state
        # self.reward[self.ptr] = reward
        # self.not_done[self.ptr] = 1.0 - float(done)

        # self.ptr += 1
        # if self.ptr >= self.max_size:
        #     self.ptr = 0
        #     self.is_full = True
            
        # self.size = self.max_size if self.is_full else self.ptr

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(np.array(reward))
        self.done.append(done)
        self.timestep.append(timestep)
        self.preference.append(prefs)
        self.next_state.append(next_state)
        self.curr_buffer_size+=1

    def save(self, filename="buffer_episodes.pkl"):
        """
        Saves the valid portion of the buffer to a compressed .npz file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.trajectories, f)

