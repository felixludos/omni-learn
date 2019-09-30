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
from torch.nn import functional as F



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
		info = {k:torch.tensor(v) for k,v in info.items()}
		return obs, reward, done, info

def gen(env, agent, drop_last_state=True, horizon=None, max_steps=None, max_episodes=None):
	#start_time = time.time()

	states = []
	actions = []
	rewards = []
	log_probs = []

	max_steps = N_STEPS
	max_episodes = None
	total_steps = 0
	total_episodes = 0

	assert not (max_episodes is None and max_steps is None), 'must specify limit'

	#print(agent.get_value(torch.ones(4)))

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

			s, r, done, _ = env.step(a)
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

		states.append(S)
		actions.append(A)
		log_probs.append(LP)
		rewards.append(R)

	return states, actions, log_probs, rewards#, values

def compute_returns(rewards, discount=1.):
	returns = []
	discount = GAMMA
	for R in rewards:
		G = R.clone()
		for i in range(G.size(0)-2, -1, -1):
			G[i] += discount * G[i+1]
		returns.append(G)

	return returns

class PolicyNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.net = nets.make_MLP(OBSERVATION_DIM, 2 * ACTION_DIM, hidden_dims=[8, 8], nonlin='prelu')
		self.act_dim = ACTION_DIM

		self.norm = RunningMeanStd(OBSERVATION_DIM)

	def get_pi(self, obs):
		obs = self.norm(obs.view(-1,OBSERVATION_DIM))
		mu, log_sigma = self(obs)
		return Normal(mu, log_sigma.exp())

	def get_action(self, obs):
		return self.get_pi(obs).sample()

	def forward(self, x):
		x = self.norm(x)
		y = self.net(x)

		return y.narrow(-1, 0, self.act_dim), y.narrow(-1, self.act_dim, self.act_dim)

class Baseline(nn.Module):
	def __init__(self, ):
		super().__init__()

		self.net = nets.make_MLP(OBSERVATION_DIM, 1, hidden_dims=[8, 8], nonlin='prelu')
		self.norm = RunningMeanStd(OBSERVATION_DIM)

		self.optim = optim.Adam(self.net.parameters(), lr=LEARNING_RATE, eps=1e-5)

		n_batch = MAX_STEPS // (N_STEPS * N_ENVS)

		self.scheduler = torch.optim.lr_scheduler.LambdaLR(
			self.optim,
			lambda ep: (n_batch - ep) / float(n_batch),
			-1)

	def train_step(self, states, returns):

		value_net_losses = []

		self.scheduler.step()

		dataloader = DataLoader(TensorDataset(states, returns),
		                        batch_size=PPO_BATCH, shuffle=True, num_workers=0)

		for ppo_iter in range(PPO_EPOCH):
			for idx, (x, y) in enumerate(dataloader):
				self.optim.zero_grad()
				values = self(x)
				loss = F.mse_loss(values, y)
				loss.backward()

				self.optim.step()

				value_net_losses.append(loss.item())

		return value_net_losses

	def forward(self, x):
		return self.net(self.norm(x))

class PPO(nn.Module):
	def __init__(self, ):
		super().__init__()

		self.policy = PolicyNet()
		self.baseline = Baseline()

		n_batch = MAX_STEPS // (N_STEPS * N_ENVS)

		self.optim = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(
								self.optim,
								lambda ep: (n_batch - ep) / float(n_batch),
								-1)

	def forward(self, x):
		pi = self.policy.get_pi(obs)
		action = pi.sample()
		return action

	def get_action(self):
		return self.policy.get_action(obs)

	def get_pi(self, x):
		return self.policy.get_pi(x)

	def train_step(self, states, actions, log_probs, returns):

		self.scheduler.step()
		policy_net_losses = []

		advantages = returns - self.baseline(states).detach()
		advantages = (advantages - advantages.mean()) / advantages.std()

		self.policy.norm.update(states)
		self.baseline.norm.update(states)

		dataset = TensorDataset(states, actions, advantages, log_probs)
		dataloader = DataLoader(dataset, batch_size=PPO_BATCH, shuffle=True, drop_last=True, num_workers=0)

		for ppo_iter in range(PPO_EPOCH):

			iter_time = time.time()

			for idx, (sample_obs, sample_actions, sample_adv, sample_action_logprobs) in enumerate(
					dataloader):
				# value_net_opt.zero_grad()
				self.optim.zero_grad()

				# (mu, logstddev), values = agent(sample_obs)
				# action_logprobs = diagmvn_logprob(mu, logstddev, sample_actions).unsqueeze(1)

				pi = agent.get_pi(sample_obs)

				action_logprobs = pi.log_prob(sample_actions)

				r = torch.exp(action_logprobs - sample_action_logprobs)

				surrogate = torch.min(r * sample_adv, r.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * sample_adv)

				# value_net_loss = torch.sum(torch.pow(sample_rewards - values, 2), dim=1)
				# value_net_loss = torch.mean(value_net_loss)
				# value_net_loss.backward()

				policy_net_loss = -surrogate
				policy_net_loss = torch.mean(policy_net_loss)
				policy_net_loss.backward()

				self.optim.step()
				# value_net_opt.step()

				policy_net_losses.append(policy_net_loss.cpu().item())
			# value_net_losses.append(value_net_loss.cpu().item())

		baseline_stats = self.baseline.train_step(states, returns)

		return policy_net_losses, baseline_stats

def train(env, agent):

	plt.ion()
	fig, ax = plt.subplots()

	n_batch = MAX_STEPS // (N_STEPS * N_ENVS)

	last_log_time = time.time()

	total_steps = 0

	from collections import deque
	history_buffer = deque([], maxlen=HISTORY_LEN)
	history = []

	def compute_history_stats(itr):
		l = list(history_buffer)
		mean_score = np.mean(l)
		print('%d last %d trials max %.1f min %.1f avg %.1f' % (itr, len(l), max(l), min(l), mean_score))
		history.append((total_steps, mean_score))
		xs, ys = zip(*history)
		ax.clear()
		ax.plot(xs, ys, marker='o')
		plt.pause(0.001)

	for train_iter in range(n_batch):
		agent.to(device=ROLLOUT_DEVICE).eval()

		states, actions, log_probs, rewards = gen(env, agent)
		returns = compute_returns(rewards)
		history_buffer.extend(r.sum().item() for r in rewards)

		states = torch.cat(states).detach().view(-1, OBSERVATION_DIM)
		actions = torch.cat(actions).detach().view(-1, ACTION_DIM)
		log_probs = torch.cat(log_probs).detach().view(-1, ACTION_DIM)
		returns = torch.cat(returns).detach().view(-1, 1)

		start_time = time.time()

		agent.to(device=TRAIN_DEVICE).train()

		policy_net_losses = []
		value_net_losses = []

		stats = agent.train_step(states, actions, log_probs, returns)

		if train_iter % LOG_INTERVAL == 0:
			policy_net_losses, value_net_losses = stats
			print('%.2f sec/batch   policy %8.2f value %8.2f lr %8.6f' % (
				(time.time() - last_log_time) / LOG_INTERVAL,
				np.mean(policy_net_losses),
				np.mean(value_net_losses),
				agent.scheduler.get_lr()[0]))
				# policy_net_scheduler.get_lr()[0]))
			last_log_time = time.time()

		total_steps += len(actions)

		compute_history_stats(train_iter)

	fig.savefig('%s' % FIGURE_NAME)


if __name__ == '__main__':
	# policy_net = PolicyNet().to(device=TRAIN_DEVICE)
	# value_net = ValueNet().to(device=TRAIN_DEVICE)

	seed = 10

	#random.seed(10)
	#np.random.seed(seed)
	torch.manual_seed(seed)

	env = Pytorch_Gym_Env(ENV_NAME)
	env._env.seed(seed)

	#agent = Agent().to(device=TRAIN_DEVICE)
	agent = PPO()

	# print(next(iter(agent.policy.net.parameters())))
	# quit()

	train(env, agent)#, policy_net_opt, value_net_opt)
