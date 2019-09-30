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


ENV_NAME = sys.argv[1]
FIGURE_NAME = sys.argv[2]

TRAIN_DEVICE = torch.device('cpu')
ROLLOUT_DEVICE = torch.device('cuda')


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


class RunningMeanStd(object):
	def __init__(self, dim):
		super(RunningMeanStd, self).__init__()
		self.dim = dim
		self.mean_sum_sq = np.zeros(dim, np.float32)
		self.mu = np.zeros(dim, np.float32)
		self.n = 0

	def update(self, xs):
		for i in range(xs.shape[0]):
			x = xs[i]
			self.mu = (self.mu * self.n + x) / (self.n + 1)
			self.mean_sum_sq = (self.mean_sum_sq * self.n + x * x) / (self.n + 1)
			self.n += 1

	def mean(self):
		if self.n <= 1:
			return np.zeros((1, self.dim), np.float32)

		return self.mu.reshape((1, self.dim))

	def stddev(self):
		if self.n <= 1:
			return np.ones((1, self.dim), np.float32)

		return np.sqrt(self.mean_sum_sq - np.power(self.mu, 2))

	def normalize_with_clip(self, x, clip_min, clip_max):
		return np.clip((x - self.mean()) / self.stddev(), clip_min, clip_max)


class PolicyNet(nn.Module):
	def __init__(self):
		super(PolicyNet, self).__init__()
		self.fc1 = nn.Linear(OBSERVATION_DIM, 64)
		self.fc2 = nn.Linear(64, 64)

		self.action_mean = nn.Linear(64, ACTION_DIM)
		self.action_logstddev = nn.Linear(64, ACTION_DIM)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))

		action_mean_pred = self.action_mean(x)
		action_logstddev_pred = self.action_logstddev(x)

		return action_mean_pred, action_logstddev_pred


class ValueNet(nn.Module):
	def __init__(self):
		super(ValueNet, self).__init__()
		self.fc1 = nn.Linear(OBSERVATION_DIM, 64)
		self.fc2 = nn.Linear(64, 64)
		self.value = nn.Linear(64, 1)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		value_pred = self.value(x)

		return value_pred


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
				'%s/mujoco_worker.py' % file_dir,
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


def step_parallel_env(env, obs, policy_net, value_net, z):
	with torch.no_grad():
		obs_th = torch.as_tensor(obs, dtype=torch.float, device=ROLLOUT_DEVICE)
		action_mean, action_logstddev = policy_net(obs_th)
		values_v = value_net(obs_th)

		# Equivalent to
		#
		# torch.exp(action_logstddev) * z + action_mean
		#
		# It seems that above statement should be faster (dot product and addition), but it is actually slower than
		# the statement below.
		# This is probably due to the size of vectors being too small.
		# To make it faster, we rewrite the dot product into a matrix multiply.
		action_samples = torch.matmul(torch.diag_embed(torch.exp(action_logstddev)),
									  z.unsqueeze(2)).squeeze(2) + action_mean
		# Also equivalent to the following (almost as fast).
		# action_samples = MultivariateNormal(action_mean, torch.diag_embed(torch.pow(torch.exp(action_logstddev), 2))).sample()

		observations, rewards, dones = env.step(action_samples.cpu().numpy())
		return action_samples, observations, rewards, dones, action_mean, action_logstddev, values_v


def run_model_parallel_env(obs, policy_net, value_net, volatile):
	obs_th = torch.as_tensor(np.array(obs, np.float32), device=ROLLOUT_DEVICE)
	if volatile:
		with torch.no_grad():
			return policy_net(obs_th), value_net(obs_th)
	else:
		return policy_net(obs_th), value_net(obs_th)


def rollout(env, policy_net, value_net, current_observations, observation_rms):
	start_time = time.time()

	# Variables ending with "_th" contain pytorch tensors. Otherwise they are numpy arrays.
	batch_obs = []
	batch_actions_th = []
	batch_rewards = []
	batch_dones = []
	batch_values_th = []
	batch_action_means_th = []
	batch_action_logstddevs_th = []

	observations = current_observations

	with torch.no_grad():
		randn = torch.randn(N_STEPS, N_ENVS, ACTION_DIM, dtype=torch.float, device=ROLLOUT_DEVICE)

	for i in range(N_STEPS):
		batch_obs.append(observations)

		actions, observations, rewards, dones, action_means, action_logstddevs, values = \
			step_parallel_env(env, observation_rms.normalize_with_clip(observations, -5.0, 5.0), policy_net, value_net, randn[i])

		batch_actions_th.append(actions)
		batch_rewards.append(rewards)
		batch_dones.append(dones)
		batch_values_th.append(values)
		batch_action_means_th.append(action_means)
		batch_action_logstddevs_th.append(action_logstddevs)

	# print('rollout:', time.time() - start_time)

	return (observations,
			batch_obs,
			batch_actions_th,
			batch_rewards,
			batch_dones,
			batch_action_means_th,
			batch_action_logstddevs_th,
			batch_values_th)


def compute_advantages(batch_rewards, batch_values, batch_dones, future_rewards):
	batch_advantages = []
	batch_cum_rewards = []
	last_values = future_rewards

	advantage = np.zeros(N_ENVS, np.float32)

	for i in range(N_STEPS - 1, -1, -1):
		for j in range(N_ENVS):
			if batch_dones[i][j]:
				last_values[j] = 0.0
				advantage[j] = 0.0

		ta = advantage[0]

		#print(advantage, ta)
		#quit()

		# if np.isclose(advantage[0, 0], 0):
		# 	print(batch_rewards[i], last_values, batch_values[i])
		# 	quit()

		# General Advantage Estimator
		delta = batch_rewards[i] + GAMMA * last_values - batch_values[i]
		advantage = delta + GAMMA * GAE_LAMBDA * advantage
		#print(delta, advantage)
		#quit()
		#print(advantage)
		#quit()
		# if np.isclose(advantage, 0).any():
		# 	print(delta, batch_rewards[i], last_values, batch_values[i])
		# 	quit()
		batch_advantages.append(advantage.copy())

		last_values = batch_values[i]
		cum_reward = advantage + batch_values[i]
		# if cum_reward == 1:
		# 	print(advantage, delta, batch_rewards[i], last_values, batch_values[i])
		# 	quit()
		batch_cum_rewards.append(cum_reward)

		# if ta == 0:
		# 	print(batch_advantages[-5:])
		# 	#quit()

	#print(batch_advantages)

	return batch_advantages[::-1], batch_cum_rewards[::-1]


def compute_action_logprobs(batch_actions_th, batch_action_means_th, batch_action_logstddevs_th):
	# Equivalent to
	# logprobs = []
	# for i in range(N_STEPS):
	#     distrib = MultivariateNormal(batch_action_means_th[i],
	#                                  torch.diag_embed(torch.pow(torch.exp(batch_action_logstddevs_th[i]), 2)))
	#     logprobs.append(distrib.log_prob(batch_actions_th[i]).data)

	logprobs = diagmvn_logprob(torch.stack(batch_action_means_th),
							   torch.stack(batch_action_logstddevs_th),
							   torch.stack(batch_actions_th))
	return logprobs


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

	env = ParallelEnv(N_ENVS)
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
	current_rewards = [0] * env.n_env
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
		 batch_obs,
		 batch_actions_th,
		 batch_rewards,
		 batch_dones,
		 batch_action_means_th,
		 batch_action_logstddevs_th,
		 batch_values_th) = rollout(env, policy_net, value_net, observations, obs_rms)

		# Update history stats
		for i in range(N_STEPS):
			for j in range(N_ENVS):
				if batch_dones[i][j]:
					history_buffer.append(current_rewards[j])
					current_rewards[j] = 0.0
				else:
					current_rewards[j] += batch_rewards[i][j]

		# Estimate the sum of future rewards
		_, future_rewards_th = run_model_parallel_env(
			obs_rms.normalize_with_clip(observations, -5.0, 5.0), policy_net, value_net, volatile=True)

		batch_values = torch.cat(batch_values_th, dim=0).cpu().numpy()

		batch_advantages, batch_cum_rewards = compute_advantages(
			batch_rewards, batch_values, batch_dones, future_rewards_th.cpu().numpy())

		#print(type(batch_advantages), type(batch_dones), type(batch_cum_rewards), type(batch_rewards))

		# ls = [batch_rewards, batch_dones, batch_cum_rewards, batch_advantages]
		# ls = [l[:50] for l in ls]
		# ls = np.array(ls)
		# print(ls.reshape(4,-1).T)
		# quit()

		batch_action_logprobs_old_th = compute_action_logprobs(
			batch_actions_th, batch_action_means_th, batch_action_logstddevs_th)

		batch_obs = np.array(batch_obs, np.float32).reshape((N_STEPS * N_ENVS, -1))
		obs_rms.update(batch_obs)
		# Strangely, normalize_with_clip() may return float64 arrays even if all np arrays are in float32.
		# As a workaround, explicitly cast it to float32.
		batch_obs = obs_rms.normalize_with_clip(batch_obs, -5.0, 5.0).astype(np.float32)

		batch_obs_th = torch.as_tensor(batch_obs)
		assert not batch_obs_th.requires_grad

		batch_actions_th = torch.cat(batch_actions_th, dim=0).view(N_STEPS * N_ENVS, -1)

		batch_advantages_th = torch.as_tensor(batch_advantages, dtype=torch.float32).view(N_STEPS * N_ENVS, -1)
		batch_advantages_th = (batch_advantages_th - batch_advantages_th.mean()) / batch_advantages_th.std()
		assert not batch_advantages_th.requires_grad

		batch_action_logprobs_old_th = batch_action_logprobs_old_th.view(N_STEPS * N_ENVS, -1)
		assert not batch_action_logprobs_old_th.requires_grad

		batch_cum_rewards_th = torch.as_tensor(batch_cum_rewards, dtype=torch.float32).view(N_STEPS * N_ENVS, 1)
		assert not batch_cum_rewards_th.requires_grad

		policy_net_scheduler.step()
		value_net_scheduler.step()

		policy_net_losses = []
		value_net_losses = []

		batch_obs_th = batch_obs_th.to(device=TRAIN_DEVICE)
		batch_actions_th = batch_actions_th.to(device=TRAIN_DEVICE)
		batch_advantages_th = batch_advantages_th.to(device=TRAIN_DEVICE)
		batch_action_logprobs_old_th = batch_action_logprobs_old_th.to(device=TRAIN_DEVICE)
		batch_cum_rewards_th = batch_cum_rewards_th.to(device=TRAIN_DEVICE)

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
