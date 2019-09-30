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

		if done:
			obs = self.reset()

		reward = torch.tensor(reward).float().to(self._device).view(1)
		done = torch.tensor(done).float().to(self._device).view(1)
		return obs, reward, done, info

from torch.distributions import Normal

def step_parallel_env(env, obs, policy_net, value_net, z=None):
	with torch.no_grad():
		obs_th = torch.as_tensor(obs, dtype=torch.float, device=ROLLOUT_DEVICE)
		action_mean, action_logstddev = policy_net(obs_th)
		values_v = value_net(obs_th)

		# start = time.time()

		# Equivalent to
		#
		#z = torch.randn(action_mean.shape, device=ROLLOUT_DEVICE, dtype=torch.float)
		#action_samples = torch.exp(action_logstddev) * z + action_mean
		#
		# It seems that above statement should be faster (dot product and addition), but it is actually slower than
		# the statement below.
		# This is probably due to the size of vectors being too small.
		# To make it faster, we rewrite the dot product into a matrix multiply.

		# print(action_mean.shape)
		# quit()
		# z = torch.randn(action_mean.shape, device=ROLLOUT_DEVICE, dtype=torch.float)
		#
		# action_samples = torch.matmul(torch.diag_embed(torch.exp(action_logstddev)),
		#                               z.unsqueeze(2)).squeeze(2) + action_mean
		# Also equivalent to the following (almost as fast).
		# action_samples = MultivariateNormal(action_mean, torch.diag_embed(torch.pow(torch.exp(action_logstddev), 2))).sample()

		action_samples = Normal(action_mean, action_logstddev.exp()).sample()

		# print(time.time() - start)

		observations, rewards, dones, _ = env.step(action_samples)
		return action_samples, observations, rewards, dones, action_mean, action_logstddev, values_v


def run_model_parallel_env(obs, policy_net, value_net, volatile):
	obs_th = obs #torch.as_tensor(np.array(obs, np.float32), device=ROLLOUT_DEVICE)
	if volatile:
		with torch.no_grad():
			return policy_net(obs_th), value_net(obs_th)
	else:
		return policy_net(obs_th), value_net(obs_th)


def rollout(env, policy_net, value_net, current_observations, observation_rms=None):
	start_time = time.time()

	# Variables ending with "_th" contain pytorch tensors. Otherwise they are numpy arrays.
	batch_obs = torch.zeros(N_STEPS, OBSERVATION_DIM).to(current_observations.device)
	batch_actions = torch.zeros(N_STEPS, ACTION_DIM).to(current_observations.device)
	batch_rewards = torch.zeros(N_STEPS).to(current_observations.device)
	batch_dones = torch.zeros(N_STEPS).to(current_observations.device)
	batch_values = torch.zeros(N_STEPS).to(current_observations.device)
	batch_action_means = torch.zeros(N_STEPS, ACTION_DIM).to(current_observations.device)
	batch_action_logstddevs = torch.zeros(N_STEPS, ACTION_DIM).to(current_observations.device)

	observations = current_observations

	# with torch.no_grad():
	#     randn = torch.randn(N_STEPS, N_ENVS, ACTION_DIM, dtype=torch.float, device=ROLLOUT_DEVICE)

	for i in range(N_STEPS):
		batch_obs[i] = observations

		actions, observations, rewards, dones, action_means, action_logstddevs, values = \
			step_parallel_env(env, policy_net.norm(observations), policy_net, value_net)#, randn[i])

		batch_actions[i] = actions
		batch_rewards[i] = rewards
		batch_dones[i] = dones
		batch_values[i] = values
		batch_action_means[i] = action_means
		batch_action_logstddevs[i] = action_logstddevs

	print('rollout:', time.time() - start_time)

	return (observations,
			batch_obs,
			batch_actions,
			batch_rewards,
			batch_dones,
			batch_action_means,
			batch_action_logstddevs,
			batch_values)


def compute_advantages(batch_rewards, batch_values, batch_dones, future_rewards):
	batch_advantages = torch.zeros(batch_values.shape).to(batch_values.device)
	batch_cum_rewards = torch.zeros(batch_values.shape).to(batch_values.device)

	#print(batch_values[:4], batch_rewards[:4], batch_dones[:4])
	#quit()

	last_value = future_rewards.item()
	advantage = 0

	batch_values = batch_values.view(N_STEPS)

	for i in range(N_STEPS - 1, -1, -1):
		if batch_dones[i]:
			last_value = 0.0
			advantage = 0.0

		# General Advantage Estimator
		delta = batch_rewards[i] + GAMMA * last_value - batch_values[i]
		advantage = delta + GAMMA * GAE_LAMBDA * advantage
		batch_advantages[i] = advantage

		last_value = batch_values[i]
		batch_cum_rewards[i] = advantage + batch_values[i]

	# print(batch_advantages.min(), batch_advantages.max())
	# print(torch.stack([batch_advantages, batch_cum_rewards, batch_dones], -1)[:30])
	# quit()

	return batch_advantages, batch_cum_rewards


def compute_action_logprobs(batch_actions_th, batch_action_means_th, batch_action_logstddevs_th):
	# Equivalent to
	# logprobs = []
	# for i in range(N_STEPS):
	#     distrib = MultivariateNormal(batch_action_means_th[i],
	#                                  torch.diag_embed(torch.pow(torch.exp(batch_action_logstddevs_th[i]), 2)))
	#     logprobs.append(distrib.log_prob(batch_actions_th[i]).data)

	logprobs = diagmvn_logprob(batch_action_means_th,
							   batch_action_logstddevs_th,
							   batch_actions_th)
	return logprobs


def diagmvn_logprob(x, mu, logstddev):
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

	#return - torch.sum(logstddev, dim=-1) - 0.5 * torch.sum(torch.pow(y / torch.exp(logstddev), 2), -1)
	return -k / 2 * math.log(2.0 * math.pi) - torch.sum(logstddev, dim=-1) - 0.5 * torch.sum(torch.pow(y / torch.exp(logstddev), 2), -1)


def train(policy_net, value_net, policy_net_opt, value_net_opt):
	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt

	plt.ion()
	fig, ax = plt.subplots()

	#env = ParallelEnv(N_ENVS)
	assert N_ENVS == 1, 'mp not supported'
	env = Pytorch_Gym_Env(ENV_NAME, device=ROLLOUT_DEVICE)
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
	current_rewards = 0
	history_buffer = deque([], maxlen=HISTORY_LEN)
	history = []

	#obs_rms = RunningMeanStd(OBSERVATION_DIM)

	def compute_history_stats():
		l = list(history_buffer)
		mean_score = np.mean(l)
		print('last %d trials max %.1f min %.1f avg %.1f' % (len(l), max(l), min(l), mean_score))
		history.append((total_steps, mean_score))
		xs, ys = zip(*history)
		ax.clear()
		ax.plot(xs, ys, marker='o')
		plt.pause(0.0001)

	for train_iter in range(n_batch):

		with torch.no_grad():

			policy_net.to(device=ROLLOUT_DEVICE)
			value_net.to(device=ROLLOUT_DEVICE)

			(observations,  # Next observations
			 batch_obs,
			 batch_actions,
			 batch_rewards,
			 batch_dones,
			 batch_action_means,
			 batch_action_logstddevs,
			 batch_values) = rollout(env, policy_net, value_net, observations)

			# Update history stats
			for i in range(N_STEPS):
				if batch_dones[i]:
					history_buffer.append(current_rewards)
					current_rewards = 0.0
				else:
					current_rewards += batch_rewards[i].item()

			# Estimate the sum of future rewards
			_, future_rewards_th = run_model_parallel_env(
				policy_net.norm(observations), policy_net, value_net, volatile=True)

			batch_advantages, batch_cum_rewards = compute_advantages(
				batch_rewards, batch_values, batch_dones, future_rewards_th)

			# ls = [batch_rewards, batch_dones, batch_cum_rewards, batch_advantages]
			# ls = torch.stack(ls, 1)
			# print(ls.squeeze()[:40])
			# quit()

			batch_action_logprobs_old_th = diagmvn_logprob(
				batch_actions, batch_action_means, batch_action_logstddevs)

			#batch_obs = torch.stack(batch_obs)

			policy_net.norm.update(batch_obs)
			# Strangely, normalize_with_clip() may return float64 arrays even if all np arrays are in float32.
			# As a workaround, explicitly cast it to float32.
			batch_obs = policy_net.norm(batch_obs)

			print(batch_action_logprobs_old_th.shape)
			print(batch_action_logprobs_old_th[:4])

			#batch_obs_th = torch.as_tensor(batch_obs)
			#assert not batch_obs_th.requires_grad

			#batch_actions_th = torch.cat(batch_actions_th, dim=0).view(N_STEPS * N_ENVS, -1)

			#batch_advantages_th = torch.as_tensor(batch_advantages, dtype=torch.float32).view(N_STEPS * N_ENVS, -1)
			#batch_advantages_th = (batch_advantages_th - batch_advantages_th.mean()) / batch_advantages_th.std()
			#assert not batch_advantages_th.requires_grad
			batch_advantages = batch_advantages.detach()

			batch_action_logprobs_old = batch_action_logprobs_old_th.view(N_STEPS * N_ENVS, -1).detach()
			assert not batch_action_logprobs_old_th.requires_grad

			#print(len(batch_cum_rewards), batch_cum_rewards[0].shape)
			#quit()

			#batch_cum_rewards_th = torch.as_tensor(batch_cum_rewards, dtype=torch.float32).view(N_STEPS * N_ENVS, 1).detach()
			#assert not batch_cum_rewards_th.requires_grad

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

		# print(*[x.shape for x in [batch_obs,
		#                         batch_actions,
		#                         batch_advantages,
		#                         batch_action_logprobs_old,
		#                         batch_cum_rewards]])
		# quit()

		dataset = TensorDataset(batch_obs,
								batch_actions,
								batch_advantages,
								batch_action_logprobs_old,
								batch_cum_rewards)

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
