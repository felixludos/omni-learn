import sys, os
import numpy as np
#import matplotlib.pyplot as plt
import copy
import gym
from gym import Space
#from gym.spaces import Discrete, Box
import cv2
from scipy.sparse import coo_matrix
from collections import namedtuple
from .general import Env


class Print_Env(Env): # only for testing
	def __init__(self, obs_dis=False, act_dis=False):
		self.obs_shape = (3,)
		shape = self.obs_shape
		# self.observation_space = Discrete(10) if obs_dis else Box(shape=shape, low=-np.random.rand(*shape), high=np.random.rand(*shape))
		# self.action_space = Discrete(3) if act_dis else Box(shape=shape,  low=-np.random.rand(*shape), high=np.random.rand(*shape))
		self.observation_space = Discrete(10) if obs_dis else Box(shape=shape, low=-np.random.rand(), high=np.random.rand())
		self.action_space = Discrete(3) if act_dis else Box(shape=shape,  low=-np.random.rand(), high=np.random.rand())
	
	def reset(self):
		state = np.random.rand(*self.obs_shape)
		print('internal state' ,state)
		return state
	
	def step(self, action):
		print('internal action', action)
		state = np.random.rand(*self.obs_shape)
		print('internal state', state)
		return state, np.random.rand(), True, {}


	
	def reset(self):
		obs = self._env.reset()
		if not self.dis_obs:
			obs /= self.obs_scale
		return obs
	
	def step(self, action):
		if not self.dis_act:
			action *= self.act_scale
		obs, reward, keepgoing, info = self._env.step(action)
		if not self.dis_obs:
			obs /= self.obs_scale
		return obs, reward, keepgoing, info
	
	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs)

class MultiAgent_Pos_Env(Env):
	def __init__(self, n_agents=4, n_particles=1):
		super(MultiAgent_Pos_Env, self).__init__()
		self.n_agents = n_agents
		self.n_particles = n_particles



_diamond = np.array([[0 ,0],
                     [0 ,1],
                     [1 ,0],
                     [0 ,-1],
                     [-1 ,0],
                     [-1 ,1],
                     [1 ,1],
                     [1 ,-1],
                     [-1 ,-1],
                     [0 ,2],
                     [2 ,0],
                     [0 ,-2],
                     [-2 ,0]])

class Discrete_Pos_Trap(MultiAgent_Pos_Env):
	def __init__(self, grid_side=11, obs_grid=True, expressive_rewards=False, **kwargs):
		super(Discrete_Pos_Trap, self).__init__(**kwargs)
		
		self.size = grid_side # grid side length
		
		self.obs_grid = obs_grid # makes particles indisinguishable in observation -> returns 2-channel grid
		self.expressive_rewards = expressive_rewards
		
		self.action_shape = (self.n_agents,)
		self.num_actions = len(_diamond)
		adj = 0 if self.size % 2 else 1
		if self.obs_grid:
			self.obs_shape = (self.n_agents, 2, self.siz e +adj, self.siz e +adj)
		else:
			self.obs_shape = (self.n_agents, self.n_agents - 1, 2)
		
		obs_dim = np.product(self.obs_shape)
		action_dim = np.product(self.action_shape)
		max_obs = 1 if self.obs_grid else (self.siz e +adj )/ /2
		min_obs = 0 if self.obs_grid else -(self.siz e +adj )/ /2
		max_action = len(_diamond) - 1
		
		self.action_options = len(_diamond)
		
		self.observation_space = namedtuple('o_s', ['shape', 'high', 'low'])(shape=self.obs_shape, high=max_ob s *np.ones(self.obs_shape), low=min_ob s *np.ones(self.obs_shape))
		self.action_space = namedtuple('a_s', ['shape', 'high', 'low'])(shape=self.action_shape, high=max_actio n *np.ones(self.action_shape), low=np.zeros(self.action_shape))
		self.spec = namedtuple('spec', ['timestep_limit', 'observation_dim', 'action_dim'])(400, self.observation_space.shape
			                                                                                    [0], self.action_space.shape[0])
		self.horizon = self.spec.timestep_limit
		
		self._non_diagonal = (np.eye(self.n_agents) - 1).astype(bool)
		self._a_ones = np.ones(self.n_agents - 1)
		self._p_ones = np.ones(self.n_particles)
	
	def _make_obs(self):
		
		agent_obs = self.agents[np.newaxis, :, :] - self.agents[:, np.newaxis, :] # A x A x 2
		particle_obs = self.particles[np.newaxis, :, :] - self.agents[:, np.newaxis, :] # A x P x 2
		
		# pbc - center particle pos to [size/2,size/2]
		agent_obs[agent_obs < 0] += self.size
		agent_obs += self.size // 2
		agent_obs %= self.size
		particle_obs[particle_obs < 0] += self.size
		particle_obs += self.size // 2
		particle_obs %= self.size
		
		if self.obs_grid:
			# build grid
			grid = np.zeros((self.n_agents, 2, self.size, self.size))
			for imgs, agents, particles in zip(grid, agent_obs, particle_obs):
				imgs[0] = coo_matrix((self._a_ones, agents.T), shape=self.obs_shape[-2:]).toarray().T
				imgs[1] = coo_matrix((self._p_ones, particles.T), shape=self.obs_shape[-2:]).toarray().T
			return grid # A x 2 x L x L
		
		agent_obs = agent_obs[self._non_diagonal].reshape(self.n_agents, self.n_agents - 1, -1) # A x A-1 x 2
		return np.concatenate([agent_obs, particle_obs], axis=1) # A x (A-1)+P x 2
	
	def reset(self, agents=None, particles=None):
		if agents is not None:
			assert agents.shape == (self.n_agents, 2)
			assert (agents >= 0).all() and (agents < self.size).all()
			self.agents = agents
		else:
			self.agents = np.random.randint(self.size, size=(self.n_agents ,2))
		
		if particles is not None:
			assert particles.shape == (self.n_particles, 2)
			assert (particles >= 0).all() and (particles < self.size).all()
			self.particles = particles
		else:
			self.particles = np.random.randint(self.size, size=(self.n_particles, 2))
		
		return self._make_obs()
	
	def step(self, actions): # -1 -> no action,
		
		assert (actions >= 0).all() and (actions < len(_diamond)).all(), 'Invalid actions:\n' + str(actions)
		assert len(actions) == self.n_agents, 'Requires 1 action for each agent'
		
		actions = actions.astype(int)
		
		# apply actions
		self.agents += _diamond[actions]
		self.agents %= self.size
		
		# move particles
		trapped = 0
		limited = 0
		
		for i, p in enumerate(self.particles):
			
			possible = [delta for delta in _diamond[1:5] if not (self.agents == p + delta).all(-1).any()]
			if len(possible) == 0:
				trapped += 1
			else:
				self.particles[i] += possible[np.random.randint(len(possible))]
			limited += 1 - len(possible) / 4
		
		self.particles %= self.size
		
		# compute reward
		reward = trapped / self.n_particles # normalized to [0,1]
		if self.expressive_rewards:
			reward += limited / self.n_particles
		
		obs = self._make_obs()
		
		keepgoing = True
		
		return obs, reward, keepgoing, {}
	
	# build new observations for each agent
	
	
	def render(self, size=None, onscreen=False, frame_rate=50):
		
		grid = np.zeros((self.size, self.size, 3))
		
		grid[: ,: ,0][tuple(self.agents.T)[::-1]] = 1
		grid[: ,: ,1][tuple(self.particles.T)[::-1]] = 1
		
		if not onscreen:
			return cv2.resize(grid ,(size, size))  if size is not None else grid
		
		if self.figax is None:
			plt.ion()
			fig, ax = plt.subplots()
		else:
			fig, ax = self.figax
			plt.pause(1 / frame_rate)
			ax.cla()
		ax.set_title('Grid')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.imshow(grid)
		self.figax = fig, ax
		return self.figax

class Coop_Guess_Number(Env):
	'''
	2 agent game:
	player 1: observes x ~ U(0,1), and must guess a number greater than x if invert is true, otherwise a smaller number
	player 2: observes invert ~ B(0.5), and must guess the number
	'''
	def __init__(self, two_way=True, separate_rewards=True):
		
		self.train()
		
		self.separate_rewards = separate_rewards
		self.two_way = two_way
		
		if two_way:
			self.obs_space = Continuous(shape=(2, 2))
			self.action_space = Continuous(shape=(2, 2))
		else:
			self.obs_space = Continuous(shape=(2, 2))
			self.action_space = Continuous(shape=(2,))
	
	def train(self):
		self.mode = 'train'
	def eval(self):
		self.mode = 'eval'
	
	def _make_obs(self, comms=(0 ,0)):
		
		obs = np.zeros(self.obs_space.shape)
		
		obs[0 ,0] = self.number
		if comms is not None:
			obs[0 ,1] = comms[0]
		
		obs[1 ,0] = self.invert
		if comms is not None:
			obs[1 ,1] = comms[1] if self.mode == 'train' else np.sign(comms[1])
		
		return obs
	
	def reset(self):
		self.number = np.random.rand()
		self.invert = np.random.randint(2 ) *2 - 1
		
		return self._make_obs()
	
	def step(self, actions):
		
		if self.two_way:
			signal = actions[: ,0]
			comms = actions[: ,1]
		else:
			signal = (actions[0],)
			comms = (actions[1], 0)
		
		# compute reward
		if self.invert > 0:
			r1 = signal[0] > self.number
		else:
			r1 = signal[0] < self.number
		r1 = 1 if r1 else -1
		
		r2 = 0
		if self.two_way:
			r2 = - (self.number - signal[1] )* *2
		
		reward = (r1, r2) if self.separate_rewards else r1 + r2
		
		# compute new observation
		obs = self._make_obs(comms)
		
		return obs, reward, False, {}
	
	def render(self, size, onscreen=False):
		
		ms = 'n={}, inv={}'.format(self.number, self.invert)
		
		if onscreen:
			print(ms)
		else:
			return ms


class Decay(object):
	def __init__(self, noise, cutoff, temp, max_dist):
		self.noise = noise
		self.cutoff = cutoff
		self.temp = temp
		self.eps = 1e-6 # for numerical stability
		self.end = max_dist - self.eps
	
	def __call__(self, dists): # adds noise
		if self.noise > 0:
			dists = dists + self.nois e *np.random.randn(dists.shape)
		return self._compute(dists.clip(self.eps, self.end)) # clips within epsilon
	
	def _compute(self, dists):
		raise Exception('not implemented')

class Exp_Decay(Decay):
	def __init__(self, **kwargs):
		super(Exp_Decay, self).__init__(**kwargs)
		
		m = np.exp(-self.cutoff / self.temp)
		self.coeff = 1. / (1 - m)
		self.const = - m / (1 - m)
	
	def _compute(self, dists):
		return self.coeff * np.exp(-dist s /self.temp) + self.const

def make_decay(decay_type, noise, cutoff, temp, max_dist):
	if decay_type == 'exp':
		return Exp_Decay(noise=noise, cutoff=cutoff, temp=temp, max_dist=max_dist)
	else:
		raise Exception('Unknown decay type: {}'.format(decay_type))

Discrete = namedtuple('Discrete', ['shape', 'n'])
Continuous = namedtuple('Continuous', ['shape'])

def ravel_mat(M): # only ravels first axis
	M = M.copy()
	for i, row in enumerate(M):
		M[i] = i * len(row) + row
	return M.reshape(-1)

class Static_Hunting(Env):
	def __init__(self, n_agents=2, n_prey=1, n_predators=0, top_k_obs=None,
	             obs_resolution=None, obs_noise=0, obs_cutoff=0.5, comm_channels=0, obs_decay='exp', obs_decay_temp=1,
	             action_resolution=None, action_cutoff=0.1,
	             reward_cutoff=0.25, independent_rewards=False ):
		'''
		no dynamics - actions are displacement <dx,dy>
		:param n_agents:
		:param n_prey:
		:param n_predators:
		:param obs_resolution: theta dim for each signal function
		:param obs_noise: % std noise for distance computation
		:param obs_cutoff: % of arena visible
		:param obs_decay_temp: temperature param for decay (effect depends on obs_decay type)
		:param comm_channels: number of binary communication channels the agents have available
		:param obs_decay: type of decay to use {exp, -1, quad}
		:param action_resolution: theta dim for action (None for continuous)
		:param action_cutoff: % of arena max action magnitude
		:param reward_cutoff: % of arena which produces a reward
		'''
		super(Static_Hunting, self).__init__()
		
		assert n_predators == 0, 'not implemented'
		assert n_agents > 0
		assert n_prey > 0
		
		self.A = n_agents
		self.P = n_prey
		self.X = n_predators
		
		assert action_resolution is None, 'only continuous actions for now'
		assert comm_channels == 0, 'no comms yet'
		assert not independent_rewards, 'not implemented'
		assert obs_resolution is None
		
		obs_resolution = max(n_agent s -1, n_predators, n_prey)
		
		channels = 1 + (1 if self.X > 0 else 0) + comm_channels
		self.obs_space = Continuous(shape=(self.A, channels, obs_resolution))
		self.action_space = Continuous(shape=(self.A, 2)) # <dx, dy>
		self.comm_channels = comm_channels
		self.comm_space = Continuous(shape=(self.A, comm_channels))
		
		self.L = 2 # side length
		self.obs_cutoff = self.L / 2 * obs_cutoff
		self.action_cutoff = self.L / 2 * action_cutoff
		self.reward_cutoff = self.L / 2 * reward_cutoff
		self.calc_weights = make_decay(obs_decay, noise=obs_noise, cutoff=self.obs_cutoff, temp=obs_decay_temp, max_dist=self. L /2)
		
		self._sel = np.eye(self.A, dtype=bool)
		self._desel = np.logical_not(self._sel)
	
	def reset(self):
		self.agents = np.random.rand(self.A, 2) * self.L
		self.prey = np.random.rand(self.P, 2) * self.L
		self.predators = np.random.rand(self.X, 2) * self.L
		
		return self._make_obs()
	
	def _pbc_shift(self, coords, shift=None):
		coords = coords.copy()
		if shift is not None:
			coords -= shift
		coords[coord s >self. L /2] -= self.L
		coords[coord s <-self. L /2] += self.L
		return coords
	
	def _make_obs(self, comms): # (A, C)
		# agent communication
		
		obs = np.zeros(self.obs_space.shape) # (A, C, R)
		C, R = self.obs_space.shape[1], self.obs_space.shape[2]
		
		if self.comm_channels > 0:
			pos = self.agents.reshape(1, self.A, 2) - self.agents.reshape(self.A, 1, 2) # (A, A, 2)
			pos = self._pbc_shift(pos[self._desel].reshape(self.A, self. A -1, 2)) # (A, A-1, 2)
			
			dists = np.sqrt((po s* *2).sum(-1, keepdims=True)) # (A, A-1, 1)
			# thetas = np.arctan2(pos[:,:,1], pos[:,:,0]) # (A, A-1)
			# idx = ((thetas/np.pi + 1) * self.obs_space.shape[-1] / 2).astype(int) # shape:(A, A-1) range:[0,R)
			
			weights = self.calc_weights(dists) # (A, A-1, 1)
			intercom = np.broadcast_to(comms, shape=(self.A, self.A, C))[self._desel].reshape(self.A, self. A -1, C) # (A, A-1, C)
			
			signals = weights * intercom # (A, A-1, C)
			order = ravel_mat(weights.reshape(self.A, self. A -1).argsort(1)) # sort weights for each agent # (A*(A-1))
			
			for c in np.ndindex(C): # loop over channels, setting obs -> assuming R >= A-1
				obs[: ,c ,:self. A -1] = signals[: ,: ,c].reshape(-1)[order].reshape(self.A, self. A -1)
		
		# prey interactions
		c = self.comm_channels # channel index
		
		pos = self.prey.reshape(1, self.P, 2) - self.agents.reshape(self.A, 1, 2)  # (A, P, 2)
		pos = self._pbc_shift(pos)  # (A, P, 2)
		
		dists = np.sqrt((pos ** 2).sum(-1, keepdims=True))  # (A, P, 1)
		weights = self.calc_weights(dists)  # (A, A-1, 1)
		
		order = ravel_mat(weights.reshape(self.A, self.P).argsort(1))  # sort weights for each agent # (A*(A-1))
		
		obs[:, c, :self.A - 1] = weights[:, :, c].reshape(-1)[order].reshape(self.A, self.A - 1)
		
		# predator interactions
		if self.X > 0:
			c = self.comm_channels + 1 # channel index
	
	def step(self, actions, comms=None):
		# move agents according to actions
		
		# move predators and prey
		
		# compute reward
		
		# compute obs - including comms
		pass
	
	def step_diff(self, actions, comms): # differentiable version of step (using torch.Tensor)
		pass
	
	def render(self, size, onscreen=False, show_obs=False):
		assert onscreen
		assert not show_obs

