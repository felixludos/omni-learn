
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import gym
from dm_control import suite

import foundation.util as util

# _env_names = {
# 	'cartpole':'CartPole-v1',
# 	'lunar-lander': 'LunarLander-v2'
# }
# def get_env_name(env_name):
# 	return _env_names[env_name]

class Replay_Episode(Dataset):
	def __init__(self, states, actions, rewards):
		self.states = states
		self.actions = actions
		self.rewards = rewards
		self.N = len(self.actions)

	def to(self, device):
		self.states.to(device)
		self.actions.to(device)
		self.rewards.to(device)

	def __len__(self):
		return self.N

	def __getitem__(self, idx):
		state = self.states[idx]
		action = self.actions[idx]
		reward = self.rewards[idx]
		done = torch.tensor(idx == self.N - 1).type_as(reward)
		next_state = self.states[(idx+1)%len(self.states)]

		return state, action, reward, done, next_state

class Replay_Buffer(Dataset):

	def __init__(self, max_episode_size=None, max_transition_size=None, device='cpu'):
		assert max_episode_size is not None or max_transition_size is not None
		self.max_episode_size = max_episode_size
		self.max_transition_size = max_transition_size
		self.device = device
		self.reset()

	def reset(self):
		self.buffer = []
		self.dataset = None
		self._idx = 0
		self.N = 0

	def to(self, device):
		self.device = device
		self.buffer = [episode.to(device)
		               for episode in self.buffer]

	def append_no_create(self, path):
		episode = Replay_Episode(*path)
		N = len(episode)

		if (self.max_transition_size is not None and self.max_transition_size < N + self.N) or \
				(self.max_episode_size is not None and self.max_episode_size < len(self.buffer) + 1):
			self._idx %= len(self.buffer)
			self.N -= len(self.buffer[self._idx])

		if self._idx < len(self.buffer):
			self.buffer[self._idx] = episode
		else:
			self.buffer.append(episode)

		self._idx += 1
		self.N += N

	def append(self, path):
		self.append_no_create(path)
		self.create_dataset()

	def extend(self, paths):  # state, action, reward
		for path in paths:
			self.append_no_create(path)

		self.create_dataset()

	def create_dataset(self):
		self.dataset = ConcatDataset(self.buffer)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]

class Score(object): # EMA to keep track of score

	def __init__(self, tau=0.01):
		self.val = None
		self.count = 0
		self.tau = tau

	def update(self, val):
		self.count += 1
		if self.val is None:
			self.val = val
			return
		self.val = self.tau * val + (1-self.tau)*self.val

	def update_all(self, vals):
		for v in vals:
			self.update(v)

	def __str__(self):
		return str(self.val)

def run_iteration(mode, N, agent, gen, horizon=None, render=False):
	train = mode == 'train'
	if train:
		agent.train()
	else:
		agent.eval()

	states, actions, rewards = zip(*[gen(horizon=horizon, render=render) for _ in range(N)])

	learn_stats = None
	if train:
		learn_stats = agent.learn(states, actions, rewards)

	rewards = torch.stack([r.sum() for r in rewards])

	return rewards


class Pytorch_DMC_Env(object):
	'''
	Wrapper for dm_control environments so they can easily be used with pytorch
	'''

	def __init__(self, domain_name, task_name, device=torch.device('cpu')):
		self._env = suite.load(domain_name, task_name)
		self._discrete = False
		self.max_steps = int(1 / self._env.control_timestep())
		self.action_space = self._env.action_spec()
		self.observation_space = self._env.observation_spec()
		self.act_dim = self.action_space.shape[0]
		self.obs_dim = sum(v.shape[0] for k,v in self.observation_space.items())
		self._device = device

	def reset(self):
		return self._wrap_obs(self._env.reset())

	def _wrap_obs(self, ts):
		obs = np.concatenate(list(ts.observation.values()))
		return torch.as_tensor(obs, dtype=torch.float, device=self._device).view(-1)
	def _unwrap_act(self, a):
		action = a.squeeze().detach().cpu().numpy()
		if not self._discrete:
			action = action.reshape(-1)
		elif action.ndim == 0:
			action = action[()]
		return action

	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs)

	def _wrap_policy(self, policy):
		def _apply_policy(ts):
			#print(ts)
			obs = self._wrap_obs(ts)
			a = self._unwrap_act(policy(obs))
			#print(a)
			return a
		return _apply_policy

	def view(self, policy=None):
		from dm_control import viewer
		if policy is not None:
			policy = self._wrap_policy(policy)
		viewer.launch(self._env, policy=policy)

	def to(self, device):
		self._device = device

	def step(self, action):
		action = self._unwrap_act(action)

		timestep = self._env.step(action)

		obs = self._wrap_obs(timestep)
		reward = torch.tensor(timestep.reward, dtype=torch.float, device=self._device).view(1)
		done = torch.tensor(timestep.last(), dtype=torch.float, device=self._device).view(1)
		return obs, reward, done

class Pytorch_Gym_Env(object):
	'''
	Wrapper for OpenAI-Gym environments so they can easily be used with pytorch
	'''
	
	def __init__(self, env_name, device='cpu'):
		self._env = gym.make(env_name)
		self._discrete = hasattr(self._env.action_space, 'n')
		self.spec = self._env.spec
		self.action_space = self._env.action_space
		self.observation_space = self._env.observation_space
		self._device = device
	
	def reset(self):
		return torch.from_numpy(self._env.reset()).float().to(self._device).view(-1)
	
	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs)
	
	def to(self, device):
		self._device = device
	
	def step(self, action):
		action = action.squeeze().detach().cpu().numpy()
		if not self._discrete:
			action = action.reshape(-1)
		elif action.ndim == 0:
			action = action[()]

		obs, reward, done, info = self._env.step(action)

		obs = torch.from_numpy(obs).float().to(self._device).view(-1)
		reward = torch.tensor(reward).float().to(self._device).view(1)
		done = torch.tensor(done).float().to(self._device).view(1)
		return obs, reward, done, info

class EMA_Update(torch.optim.Optimizer):

	def __init__(self, source_params, target_params, tau=0.001):
		super(EMA_Update, self).__init__([{'params': source_params},
		                                  {'params': target_params}], defaults={'tau': tau})

	def step(self):
		source_group, target_group = self.param_groups
		tau = source_group['tau']
		for source_param, target_param in zip(source_group['params'], target_group['params']):
			target_param.data.mul_(1. - tau)
			target_param.data.add_(tau, source_param.data)

class Generator(object):
	'''
	Generates rollouts of an environment using a policy
	'''
	
	def __init__(self, env, policy, max_steps=None, drop_last_state=True):
		
		self.created = 0
		
		self.policy = policy
		self.env = env
		
		self.drop_last_state = drop_last_state
		
		self.horizon = env.max_steps if max_steps is None else max_steps
	
	def __len__(self):
		return self.created
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self()
	
	def __call__(self, horizon=None, render=False):
		
		states = []
		actions = []
		rewards = []
		
		states.append(self.env.reset())
		horizon = self.horizon if horizon is None else horizon
		for _ in range(horizon):
			
			if render:
				self.env.render()
			
			actions.append(self.policy(states[-1]))

			out = self.env.step(actions[-1])
			state, reward, done = out[0], out[1], out[2]
			
			states.append(state)
			rewards.append(reward)
			
			if done:
				break
		
		if self.drop_last_state:
			states.pop()
		
		states = torch.stack(states)
		actions = torch.stack(actions)
		rewards = torch.cat(rewards)

		self.created += 1
		
		return states, actions, rewards

# Potentially useful utility functions

def compute_returns(rewards, discount):
	'''
	Computes estimate of discounted reward from a sequence of rewards and the discount factor
	:param rewards: 1D tensor of rewards for an episode
	:param discount: discount factor
	:return: returns (discounted rewards)
	'''
	returns = rewards.clone()
	for i in range(len(returns) - 2, -1, -1):
		returns[i] += discount * returns[i + 1]
	
	return returns

def MLE(distrib):
	'''
	Returns Maximum liklihood estimate for the given distribution
	:param distrib: pytorch distribution, should be an instance of one of the distributions listed below
	:return: the maximum liklihood estimate for some parameter of the distribution (eg. mode)
	'''
	if isinstance(distrib, Normal):
		return distrib.loc
	elif isinstance(distrib, Categorical):
		return distrib.probs.max(-1)[1]
	raise Exception('Distribution {} not recognized (did you forget to add it to MLE function?)'.format(type(distrib)))

def to_one_hot(idx, max_idx=None):
	if max_idx is None:
		max_idx = idx.max()
	dims = (max_idx,)
	if idx.ndimension() >= 1:
		if idx.size(-1) != 1:
			idx = idx.unsqueeze(-1)
		dims = idx.size()[:-1] + dims
	return torch.zeros(*dims).to(idx.device).scatter_(-1, idx.long(), 1)

