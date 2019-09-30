
import sys, os
import numpy as np
#import matplotlib.pyplot as plt
import copy
import torch
import gym
from gym.spaces import Discrete, Box
import cv2
from scipy.sparse import coo_matrix
from collections import namedtuple
from .. import framework as fm
from .. import util
import torch.multiprocessing as mp

def get_environment(env_name=None):
	if env_name is None: print("Need to specify environment name")
	return GymEnv(env_name)

class Space(object):
	def __init__(self, shape, dtype):
		self.shape = shape
		self.dtype = dtype

class Discrete_Space(Space):
	def __init__(self, choices, shape=(1,), dtype=int):
		super(Discrete_Space, self).__init__(shape, dtype)
		self.size = int(np.prod(self.shape))
		self.choices = choices
		
	def sample(self):
		return np.random.randint(self.choices, size=self.shape)
	
	def contains(self, x):
		return x.shape == self.shape and x.dtype == int and (self.choices - x > 0).all() and (x >= 0).all()
	
class Continuous_Space(Space):
	def __init__(self, shape=(1,), high=None, low=None, dtype=float):
		super(Continuous_Space, self).__init__(shape, dtype)
		self.size = int(np.prod(shape))
		self.high = high
		self.low = low
		self.full_high = np.broadcast_to(self.high, shape=self.shape).astype(dtype)
		self.full_low = np.broadcast_to(self.low, shape=self.shape).astype(dtype)
	
	def sample(self):
		return np.random.rand(*self.shape).astype(self.dtype)*(self.full_high-self.full_low)+self.full_low
	
	def contains(self, x):
		return x.shape == self.shape and x.dtype == self.dtype and (self.full_high - x >= 0).all() and (self.full_low - x <= 0).all()

class EnvSpec(object): # Only for 1 agent, for multi agent, using 1 spec per agent (with fm.Multi_Agent_Env).
	def __init__(self, obs_space, act_space, horizon, reward_space=None):
		self.obs_space = format_space(obs_space)
		self.action_space = format_space(act_space)
		self.reward_space = format_space(reward_space)
		self.horizon = horizon

def format_space(space):
	if space is None:
		return None
	elif isinstance(space, Space):
		return space
	elif isinstance(space, Discrete):
		return Discrete_Space(choices=space.n)
	elif isinstance(space, Box):
		return Continuous_Space(shape=space.shape, low=space.low, high=space.high)
	else:
		raise Exception('Space not recognized: {}'.format(space))


class Multi_Agent_Env(fm.Env):  # specs is a list of EnvSpec, 1 for each agent (cmd first, if exists)
	def __init__(self, specs, ID=None):
		super(Multi_Agent_Env, self).__init__(specs[0], ID)
		self.specs = specs

def make_env_wrapper(env_type):

	class Env_Wrapper(env_type):

		def _to_np(self, a):
			return a.detach().cpu().numpy()

		def step(self, action):

			action = self._to_np(action)

			obs, reward, done, info = super(Env_Wrapper, self).step(action)

			obs, reward = torch.tensor(obs).float().squeeze(), torch.tensor(reward).float().squeeze()

			return obs, reward, done, info

		def reset(self, init_state=None):

			if init_state is not None:
				init_state = self._to_np(init_state)

			obs = super(Env_Wrapper, self).reset(init_state)

			return torch.tensor(obs).float().squeeze()

		def render(self, *args, **kwargs):

			img = super(Env_Wrapper, self).render(*args, **kwargs)

			try:
				img = torch.tensor(img)
			except:
				img = (torch.tensor(i) for i in img)

			return img

	return Env_Wrapper

class GymEnv(fm.Env):
	def __init__(self, env_name):
		env = gym.make(env_name)
		self.env = env
		
		self._horizon = env.spec.timestep_limit
		
		try:
			self._action_dim = self.env.env.action_dim
		except AttributeError:
			if isinstance(self.env.env.action_space, Box):
				self._action_dim = self.env.env.action_space.shape[0]
			else:
				self._action_dim = self.env.env.action_space.n
		
		try:
			self._observation_dim = self.env.env.obs_dim
		except AttributeError:
			self._observation_dim = self.env.env.observation_space.shape[0]
		
		try:
			self._num_agents = self.env.env.num_agents
		except AttributeError:
			self._num_agents = 1
		
		super(GymEnv, self).__init__(spec=EnvSpec(self.env.observation_space, self.env.action_space, horizon=self._horizon),
		                             ID=env.spec.id)
	
	@property
	def action_dim(self):
		return self._action_dim
	
	@property
	def observation_dim(self):
		return self._observation_dim
	
	@property
	def observation_space(self):
		return self._observation_space
	
	@property
	def action_space(self):
		return self._action_space
	
	@property
	def horizon(self):
		return self._horizon
	
	def seed(self, seed=None):
		self.env.seed(seed)
	
	def reset(self, init_state=None):
		return self.env.reset()
	
	def step(self, action):
		return self.env.step(action)
	
	def render(self, *args, **kwargs):
		self.env.render(*args, **kwargs)
	
	def visualize_policy(self, N, policy, T=None):
		if T is None:
			T = self._horizon
			
		try:
			#print('trying native')
			self.env.env.visualize_policy(policy, T, N)
		except:
			#print('native failed')
			super(GymEnv, self).visualize_policy(N, policy, T=T)