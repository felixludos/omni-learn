
import sys, os, time
import numpy as np
import torch
import torch.multiprocessing as mp
from .. import util

import gym

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
		self._dtype = torch.float

	def reset(self):
		return torch.as_tensor(self._env.reset().copy(), dtype=torch.float, device=self._device).view(-1)

	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs)

	def to(self, device):
		self._device = device

	def seed(self, seed=None):
		if seed is None:
			seed = util.get_random_seed()
		seed = int(np.abs(seed))
		self._env.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		return seed

	def step(self, action):
		action = action.squeeze().detach().cpu().numpy()
		# action = action
		if not self._discrete:
			action = action.reshape(-1)
		elif action.ndim == 0:
			action = action[()]

		obs, reward, done, info = self._env.step(action)

		obs = torch.as_tensor(obs.copy(), dtype=torch.float, device=self._device).view(-1)
		reward = torch.as_tensor(reward, dtype=torch.float, device=self._device).view(1)
		done = torch.as_tensor(done, dtype=torch.float, device=self._device).view(1)
		info = {k:torch.as_tensor(v, dtype=torch.float, device=self._device) for k,v in info.items()}
		return obs, reward, done, info

try:
	from dm_control import suite
except ModuleNotFoundError:
	print('Failed to import dm_control')

class Pytorch_DMC_Env(object):
	'''
	Wrapper for dm_control environments so they can easily be used with pytorch
	'''

	def __init__(self, domain_name, task_name, seed=None, device=torch.device('cpu')):
		kwargs = None# if seed is None else {'seed':seed}
		self._env = suite.load(domain_name, task_name, task_kwargs=kwargs)
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


