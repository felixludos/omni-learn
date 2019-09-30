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

		self.norm = RunningMeanStd(OBSERVATION_DIM)

	def forward(self, x):
		y = self.net(x)

		return y.narrow(-1, 0, self.act_dim), y.narrow(-1, self.act_dim, self.act_dim)

class ValueNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nets.make_MLP(OBSERVATION_DIM, 1, hidden_dims=[8, 8], nonlin='prelu')

	def forward(self, x):
		return self.net(x)


class ParallelEnv(object):
	def __init__(self, n_env):
		self.n_env = n_env
		file_dir = os.path.dirname(os.path.abspath(__file__))
		procs = []
		addrs = []

		for i in range(n_env):
			addr = 'ipc:///tmp/%s.%d.%f' % (ENV_NAME, i, time.time())
			addrs.append(addr)
			procs.append(subprocess.Popen([
				sys.executable,
				'%s/env_worker.py' % file_dir,
				addrs[i],
				ENV_NAME
			]))

		sockets = []
		self.context = zmq.Context()
		for i in range(n_env):
			socket = self.context.socket(zmq.REQ)
			while True:
				try:
					socket.connect(addrs[i])
					break
				except:
					time.sleep(1.0)

			print('connected to ', addrs[i])
			sockets.append(socket)

		self.procs = procs
		self.addrs = addrs
		self.sockets = sockets

	def reset(self):
		for sock in self.sockets:
			sock.send_pyobj(['reset'], flags=zmq.NOBLOCK)

		observations = []
		for sock in self.sockets:
			observations.append(sock.recv_pyobj())
		return observations

	def step(self, actions):
		observations = []
		rewards = []
		dones = []

		for i in range(len(actions)):
			self.sockets[i].send_pyobj(['step', actions[i]], flags=zmq.NOBLOCK)

		for i in range(len(actions)):
			ob, reward, done = self.sockets[i].recv_pyobj()
			observations.append(ob)
			rewards.append(reward)
			dones.append(done)

		return observations, rewards, dones

	def __del__(self):
		for sock in self.sockets:
			sock.send_pyobj(['exit'], flags=zmq.NOBLOCK)


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
		#return self._env.reset()
		return torch.from_numpy(self._env.reset()).float().to(self._device).view(-1)
	
	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs)
	
	def to(self, device):
		self._device = device
	
	def step(self, action):
		action = action.squeeze().detach().cpu().numpy()
		#action = action
		if not self._discrete:
			action = action.reshape(-1)
		elif action.ndim == 0:
			action = action[()]
		
		obs, reward, done, info = self._env.step(action)
		
		if done:
			obs = self._env.reset()
		
		#return obs, reward, done
		
		obs = torch.from_numpy(obs).float().to(self._device).view(-1)
		reward = torch.tensor(reward).float().to(self._device).view(1)
		done = torch.tensor(done).float().to(self._device).view(1)
		return obs, reward, done
		

from torch.distributions import Normal
def step_parallel_env(env, obs_th, policy_net, value_net):
	with torch.no_grad():
		action_mean, action_logstddev = policy_net(obs_th)
		values_v = value_net(obs_th)
		
		action_samples = Normal(action_mean, action_logstddev.exp()).sample()

		observations, rewards, dones = env.step(action_samples)
		return action_samples, observations, rewards, dones, action_mean, action_logstddev, values_v


def run_model_parallel_env(obs_th, policy_net, value_net, volatile):
	if volatile:
		with torch.no_grad():
			return policy_net(obs_th), value_net(obs_th)
	else:
		return policy_net(obs_th), value_net(obs_th)


def rollout(env, policy_net, value_net, current_observations, observation_rms):
	start_time = time.time()
	
	batch_obs = torch.zeros(N_STEPS, OBSERVATION_DIM)
	batch_actions_th = torch.zeros(N_STEPS, ACTION_DIM)
	batch_rewards = torch.zeros(N_STEPS, 1)
	batch_dones = torch.zeros(N_STEPS, 1)
	batch_values_th = torch.zeros(N_STEPS, 1)
	batch_action_means_th = torch.zeros(N_STEPS, ACTION_DIM)
	batch_action_logstddevs_th = torch.zeros(N_STEPS, ACTION_DIM)

	observations = current_observations

	for i in range(N_STEPS):
		batch_obs[i] = observations

		actions, observations, rewards, dones, action_means, action_logstddevs, values = \
			step_parallel_env(env, observation_rms(observations), policy_net, value_net)

		batch_actions_th[i] = (actions)
		batch_rewards[i] = (rewards)
		batch_dones[i] = (dones)
		batch_values_th[i] = (values)
		batch_action_means_th[i] = (action_means)
		batch_action_logstddevs_th[i] = (action_logstddevs)

	print('rollout:', time.time() - start_time)

	return (observations,
			batch_obs,
			batch_actions_th,
			batch_rewards,
			batch_dones,
			batch_action_means_th,
			batch_action_logstddevs_th,
			batch_values_th)


def compute_advantages(batch_rewards, batch_values, batch_dones, last_values):
	
	batch_advantages = torch.zeros(*batch_rewards.shape)
	batch_cum_rewards = torch.zeros(*batch_rewards.shape)

	advantage = 0

	for i in range(N_STEPS - 1, -1, -1):
		if batch_dones[i]:
			last_values = 0.0
			advantage = 0.0
		
		# General Advantage Estimator
		delta = batch_rewards[i] + GAMMA * last_values - batch_values[i]
		advantage = delta + GAMMA * GAE_LAMBDA * advantage

		batch_advantages[i] = advantage.clone()

		last_values = batch_values[i]
		cum_reward = advantage + batch_values[i]
		batch_cum_rewards[i] = cum_reward

	return batch_advantages, batch_cum_rewards

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


def train(policy_net, value_net, policy_net_opt, value_net_opt):
	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt

	plt.ion()
	fig, ax = plt.subplots()

	env = Pytorch_Gym_Env(ENV_NAME)
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
	current_rewards = 0 #* env.n_env
	history_buffer = deque([], maxlen=HISTORY_LEN)
	history = []

	obs_rms = RunningMeanStd(OBSERVATION_DIM)

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
		policy_net.to(device=ROLLOUT_DEVICE)
		value_net.to(device=ROLLOUT_DEVICE)

		(observations,  # Next observations
		 batch_obs_th,
		 batch_actions_th,
		 batch_rewards_th,
		 batch_dones_th,
		 batch_action_means_th,
		 batch_action_logstddevs_th,
		 batch_values_th) = rollout(env, policy_net, value_net, observations, obs_rms)
		
		# Update history stats
		for i in range(N_STEPS):
			if batch_dones_th[i]:
				history_buffer.append(current_rewards)
				current_rewards = 0.0
			else:
				current_rewards += batch_rewards_th[i].item()
				
		# Estimate the sum of future rewards
		_, future_rewards_th = run_model_parallel_env(
			obs_rms(observations), policy_net, value_net, volatile=True)

		batch_advantages_th, batch_cum_rewards_th = compute_advantages(
			batch_rewards_th, batch_values_th, batch_dones_th, future_rewards_th.squeeze())

		batch_action_logprobs_old_th = diagmvn_logprob(
								batch_action_means_th,
							    batch_action_logstddevs_th,
							    batch_actions_th).view(-1, 1)

		obs_rms.update(batch_obs_th)
		batch_obs_th = obs_rms(batch_obs_th)
		
		batch_advantages_th = (batch_advantages_th - batch_advantages_th.mean()) / batch_advantages_th.std()

		assert not batch_obs_th.requires_grad
		assert not batch_advantages_th.requires_grad
		assert not batch_action_logprobs_old_th.requires_grad
		assert not batch_cum_rewards_th.requires_grad

		policy_net_scheduler.step()
		value_net_scheduler.step()

		policy_net_losses = []
		value_net_losses = []

		#batch_obs_th = batch_obs_th.to(device=TRAIN_DEVICE)
		#batch_actions_th = batch_actions_th.to(device=TRAIN_DEVICE)
		#batch_advantages_th = batch_advantages_th.to(device=TRAIN_DEVICE)
		#batch_action_logprobs_old_th = batch_action_logprobs_old_th.to(device=TRAIN_DEVICE)
		#batch_cum_rewards_th = batch_cum_rewards_th.to(device=TRAIN_DEVICE)

		start_time = time.time()

		policy_net.to(device=TRAIN_DEVICE)
		value_net.to(device=TRAIN_DEVICE)

		dataset = TensorDataset(batch_obs_th,
								batch_actions_th,
								batch_advantages_th,
								batch_action_logprobs_old_th,
								batch_cum_rewards_th)

		for ppo_iter in range(PPO_EPOCH):
			dataloader = DataLoader(dataset, batch_size=PPO_BATCH, shuffle=True, drop_last=True, num_workers=0)

			iter_time = time.time()

			for idx, (sample_obs, sample_actions, sample_adv, sample_action_logprobs, sample_rewards) in enumerate(dataloader):
				value_net_opt.zero_grad()
				policy_net_opt.zero_grad()

				mu, logstddev = policy_net(sample_obs)
				values = value_net(sample_obs)

				# Equivalent to
				# MultivariateNormal(mu, torch.diag_embed(torch.pow(torch.exp(logstddev), 2))).log_prob(sample_actions).unsqueeze(1)
				# but 3X faster.
				action_logprobs = diagmvn_logprob(mu, logstddev, sample_actions).unsqueeze(1)

				# print(action_logprobs.size(), sample_action_logprobs.size())

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

		total_steps += N_STEPS

		compute_history_stats()

	fig.savefig('%s' % FIGURE_NAME)


if __name__ == '__main__':
	policy_net = PolicyNet().to(device=TRAIN_DEVICE)
	value_net = ValueNet().to(device=TRAIN_DEVICE)
	policy_net_opt = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, eps=1e-5)
	value_net_opt = optim.Adam(value_net.parameters(), lr=LEARNING_RATE, eps=1e-5)
	train(policy_net, value_net, policy_net_opt, value_net_opt)
