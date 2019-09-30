import torch
from torch.distributions import MultivariateNormal
import math
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import gym
import numpy as np
import time
import subprocess
import os
import zmq
import sys
import foundation as fd
from foundation import nets
import matplotlib.pyplot as plt
from torch.distributions import Normal

ENV_NAME = sys.argv[1]
FIGURE_NAME = sys.argv[2]

TRAIN_DEVICE = torch.device('cpu')
ROLLOUT_DEVICE = torch.device('cpu')


def get_env_dimensions():
	env = gym.make(ENV_NAME)
	return len(env.observation_space.low), len(env.action_space.low)


OBSERVATION_DIM, ACTION_DIM = get_env_dimensions()


PPO_CLIP = 0.2
PPO_EPOCH = 10
PPO_BATCH = 64

GAE_LAMBDA = 0.95
GAMMA = 0.99
LEARNING_RATE = 3e-4
N_ENVS = 1
N_STEPS = 2048
MAX_STEPS = 10 ** 6
LOG_INTERVAL = 10
HISTORY_LEN = 100


class RunningMeanStd(nn.Module):
	def __init__(self, dim, cmin=-5, cmax=5):
		super().__init__()
		self.dim = dim
		self.n = 0
		self.cmin, self.cmax = cmin, cmax

		self.register_buffer('sum_sq', torch.zeros(dim))
		self.register_buffer('sum', torch.zeros(dim))
		self.register_buffer('mu', torch.zeros(dim))
		self.register_buffer('sigma', torch.ones(dim))

	def update(self, xs):
		xs = xs.view(-1, self.dim)
		self.n += xs.shape[0]
		self.sum += xs.sum(0)
		self.mu = self.sum / self.n

		self.sum_sq += xs.pow(2).sum(0)
		self.mean_sum_sq = self.sum_sq / self.n

		if self.n > 1:
			self.sigma = (self.mean_sum_sq - self.mu**2).sqrt()

	def forward(self, x):
		return ((x - self.mu) / self.sigma).clamp(self.cmin, self.cmax)

class PolicyNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.net = nets.make_MLP(OBSERVATION_DIM, 2 * ACTION_DIM, hidden_dims=[8, 8], nonlin='prelu')
		self.act_dim = ACTION_DIM

		#self.norm = RunningMeanStd(OBSERVATION_DIM)

	def forward(self, x):
		y = self.net(x)

		return y.narrow(-1, 0, self.act_dim), y.narrow(-1, self.act_dim, self.act_dim)

class ValueNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nets.make_MLP(OBSERVATION_DIM, 1, hidden_dims=[8, 8], nonlin='prelu')

	def forward(self, x):
		return self.net(x)

class Agent(nn.Module):

	def __init__(self):
		super().__init__()
		self.policy = PolicyNet()
		self.value = ValueNet()

		self.norm = RunningMeanStd(OBSERVATION_DIM)

	def forward(self, obs):
		obs = self.norm(obs)
		return self.policy(obs), self.value(obs)

	def get_pi(self, obs):
		obs = self.norm(obs)
		mu, log_sigma = self.policy(obs)
		return Normal(mu, log_sigma.exp())

	def get_action(self, obs):
		obs = self.norm(obs)
		return self.policy(obs)

	def get_value(self, obs):
		obs = self.norm(obs)
		return self.value(obs)

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
		# return self._env.reset()
		return torch.from_numpy(self._env.reset()).float().to(self._device).view(-1)

	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs)

	def to(self, device):
		self._device = device

	def step(self, action):
		action = action.squeeze().detach().cpu().numpy()
		# action = action
		if not self._discrete:
			action = action.reshape(-1)
		elif action.ndim == 0:
			action = action[()]

		obs, reward, done, info = self._env.step(action)

		obs = torch.from_numpy(obs).float().to(self._device).view(-1)
		reward = torch.tensor(reward).float().to(self._device).view(1)
		done = torch.tensor(done).float().to(self._device).view(1)
		return obs, reward, done

def gen(env, agent, drop_last_state=True, horizon=None, max_steps=None, max_episodes=None):
	start_time = time.time()

	states = []
	actions = []
	rewards = []
	values = []
	log_probs = []

	max_steps = N_STEPS
	max_episodes = None
	total_steps = 0
	total_episodes = 0

	assert not (max_episodes is None and max_steps is None), 'must specify limit'

	while (max_episodes is None or total_episodes < max_episodes) \
			and (max_steps is None or total_steps < max_steps):

		S = []
		A = []
		LP = []
		R = []

		S.append(env.reset())
		total_episodes += 1

		done = False
		steps = 0
		while not done and (horizon is None or steps < horizon):
			pi = agent.get_pi(S[-1])
			a = pi.sample()

			lp = pi.log_prob(a)
			A.append(a)
			LP.append(lp)

			s, r, done = env.step(a)
			S.append(s)
			R.append(r)

			steps += 1
		total_steps += steps

		if drop_last_state:
			S = S[:-1]

		S = torch.stack(S)
		A = torch.stack(A)
		LP = torch.stack(LP).detach()
		R = torch.stack(R)
		V = agent.value(S).detach()

		states.append(S)
		actions.append(A)
		log_probs.append(LP)
		rewards.append(R)
		values.append(V)

	return states, actions, log_probs, rewards, values

def compute_returns(rewards, discount=1.):
	returns = []
	discount = GAMMA
	for R in rewards:
		G = R.clone()
		for i in range(G.size(0)-2, -1, -1):
			G[i] += discount * G[i+1]
		returns.append(G)

	return returns

def compute_separate_advantages(rewards, values, discount=1., gae_lambda=1.):
	advantages = []
	returns = []

	discount = GAMMA
	gae_lambda = GAE_LAMBDA

	for R, V in zip(rewards, values):

		deltas = R + discount * V[1:] - V[:-1]
		G = R.clone()

		factor = discount * gae_lambda
		for i in range(deltas.size(0)-2, -1, -1):
			deltas[i] += factor * deltas[i+1]
			G[i] += discount * G[i+1]

		advantages.append(deltas)
		returns.append(G)

	return advantages, returns

def diagmvn_logprob(mu, logstddev, x):
	"""
	Compute log probability of a multivariate normal distribution with a diagonal covariance matrix.
	:param mu: (N x DIM)
	:param logstddev: (N x DIM)
	:param x: (N x DIM)
	:return: (N x 1)
	"""
	assert mu.size() == logstddev.size()
	assert mu.size() == x.size()
	k = mu.size(-1)
	y = x - mu
	return -k / 2 * math.log(2.0 * math.pi) - torch.sum(logstddev, dim=-1) - 0.5 * torch.sum(torch.pow(y / torch.exp(logstddev), 2), -1)


def train(agent, policy_net_opt, value_net_opt):

	plt.ion()
	fig, ax = plt.subplots()

	env = Pytorch_Gym_Env(ENV_NAME)
	#env = Pytorch_Gym_Env_Repeater(ENV_NAME)
	observations = env.reset()

	n_batch = MAX_STEPS // (N_STEPS * N_ENVS)

	# Linear decay
	policy_net_scheduler = torch.optim.lr_scheduler.LambdaLR(
		policy_net_opt,
		lambda ep: (n_batch - ep) / float(n_batch),
		-1)
	value_net_scheduler = torch.optim.lr_scheduler.LambdaLR(
		value_net_opt,
		lambda ep: (n_batch - ep) / float(n_batch),
		-1)

	last_log_time = time.time()

	total_steps = 0

	from collections import deque
	history_buffer = deque([], maxlen=HISTORY_LEN)
	history = []

	def compute_history_stats():
		l = list(history_buffer)
		mean_score = np.mean(l)
		print('last %d trials max %.1f min %.1f avg %.1f' % (len(l), max(l), min(l), mean_score))
		history.append((total_steps, mean_score))
		xs, ys = zip(*history)
		ax.clear()
		ax.plot(xs, ys, marker='o')
		plt.pause(0.001)

	for train_iter in range(n_batch):
		agent.to(device=ROLLOUT_DEVICE).eval()

		states, actions, log_probs, rewards, values = gen(env, agent)

		#advantages, returns = compute_separate_advantages(rewards, values)
		returns = compute_returns(rewards)

		history_buffer.extend(r.sum() for r in rewards)

		states = torch.cat(states).detach()
		actions = torch.cat(actions).detach()
		log_probs = torch.cat(log_probs).detach()
		returns = torch.cat(returns).detach()
		values = torch.cat(values).detach()

		#advantages = torch.cat(advantages).detach()
		advantages = returns - values
		# Batch normalizing advantages
		advantages = (advantages - advantages.mean()) / advantages.std()

		# Update state normalization
		agent.norm.update(states)

		# LR decay
		policy_net_scheduler.step()
		value_net_scheduler.step()

		policy_net_losses = []
		value_net_losses = []

		start_time = time.time()

		agent.to(device=TRAIN_DEVICE).train()

		dataset = TensorDataset(states, actions, advantages, log_probs, returns)

		for ppo_iter in range(PPO_EPOCH):
			dataloader = DataLoader(dataset, batch_size=PPO_BATCH, shuffle=True, drop_last=True, num_workers=0)

			iter_time = time.time()

			for idx, (sample_obs, sample_actions, sample_adv, sample_action_logprobs, sample_rewards) in enumerate(dataloader):
				value_net_opt.zero_grad()
				policy_net_opt.zero_grad()

				(mu, logstddev), values = agent(sample_obs)

				action_logprobs = diagmvn_logprob(mu, logstddev, sample_actions).unsqueeze(1)

				r = torch.exp(action_logprobs - sample_action_logprobs)

				surrogate = torch.min(r * sample_adv, r.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * sample_adv)

				value_net_loss = torch.sum(torch.pow(sample_rewards - values, 2), dim=1)
				value_net_loss = torch.mean(value_net_loss)
				value_net_loss.backward()

				policy_net_loss = -surrogate
				policy_net_loss = torch.mean(policy_net_loss)
				policy_net_loss.backward()

				policy_net_opt.step()
				value_net_opt.step()

				if train_iter % LOG_INTERVAL == 0:
					policy_net_losses.append(policy_net_loss.cpu().item())
					value_net_losses.append(value_net_loss.cpu().item())

			# print('iter time:', time.time() - iter_time)

		# print('ppo time:', time.time() - start_time)

		if train_iter % LOG_INTERVAL == 0:
			print('%.2f sec/batch   policy %8.2f value %8.2f lr %8.6f' % (
				(time.time() - last_log_time) / LOG_INTERVAL,
				np.mean(policy_net_losses),
				np.mean(value_net_losses),
				policy_net_scheduler.get_lr()[0]))
			last_log_time = time.time()

		total_steps += len(dataset)

		compute_history_stats()

	fig.savefig('%s' % FIGURE_NAME)


if __name__ == '__main__':
	# policy_net = PolicyNet().to(device=TRAIN_DEVICE)
	# value_net = ValueNet().to(device=TRAIN_DEVICE)

	agent = Agent().to(device=TRAIN_DEVICE)


	policy_net_opt = optim.Adam(agent.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
	value_net_opt = optim.Adam(agent.value.parameters(), lr=LEARNING_RATE, eps=1e-5)
	train(agent, policy_net_opt, value_net_opt)
