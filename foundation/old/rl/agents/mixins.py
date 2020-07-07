

import sys, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from foundation import framework as fm
from foundation import models
from foundation import util

####################
# Mixins
####################

class Baseline_Agent(fm.Agent): # Not Q Function (only takes state as input)
	def __init__(self, baseline, **other):
		super().__init__(**other)

		self.baseline = baseline
		if hasattr(self.baseline, 'stats'):
			self.stats.shallow_join(self.baseline.stats, 'bsln-')

		assert self.baseline.din == self.state_dim, 'policy and baseline dont have the same state space: {} vs {}'.format(
			self.baseline.din, self.state_dim)

	def learn(self, **paths):
		paths = super().learn(**paths)

		if 'values' in paths:
			values = paths['values']
		elif 'returns' in paths:
			values = paths['returns']
		else:
			values = paths['rewards']

		self.baseline.learn(paths['states'], values)
		return paths

class Discounted_Agent(fm.Agent):
	def __init__(self, discount=0.99, **other):
		super().__init__(**other)
		self.discount = discount

	def compute_returns(self, rewards, **paths):
		if isinstance(rewards, torch.Tensor):
			rewards = rewards.permute(1, 0, 2)  # iterate over timesteps

			returns = rewards.clone()
			for i in range(returns.size(0) - 2, -1, -1):
				returns[i] += self.discount * returns[i + 1]

			return returns.permute(1, 0, 2)  # switch back to episodes first

		returns = []
		for R in rewards:
			G = R.clone()
			for i in range(G.size(0) - 2, -1, -1):
				G[i] += self.discount * G[i + 1]
			returns.append(G)

		return returns

	def _format_paths(self, **paths):

		if 'returns' not in paths:
			paths['returns'] = self.compute_returns(**paths)
		if not isinstance(paths['returns'], torch.Tensor):
			paths['returns'] = torch.cat(paths['returns'])
		paths['returns'] = paths['returns'].contiguous().view(-1, 1)

		return super()._format_paths(**paths)

class Differentiable_Agent(fm.Agent):

	def __init__(self, optim=None, scheduler=None,
	             optim_type='adam', lr=1e-3, weight_decay=None, scheduler_lin=None,
	             batch_size=64, epochs_per_step=10, **other):
		super().__init__(**other)

		self.batch_size = batch_size
		self.epochs = epochs_per_step

		self.optim = optim
		if self.optim is None:
			self.optim = util.get_optimizer(optim_type, self.policy.parameters(),
			                                lr=lr, weight_decay=weight_decay)

		self.scheduler = scheduler
		if self.scheduler is None and scheduler_lin is not None:
			self.scheduler = torch.optim.lr_scheduler.LambdaLR(
				self.optim, lambda x: (scheduler_lin - x) / scheduler_lin, -1)

		if self.scheduler is not None:
			self.stats.new('lr')


class Advantage_Agent(Baseline_Agent, Discounted_Agent):

	def compute_advantages(self, states, returns, **other):
		if self.baseline is None:
			return returns.clone()
		return returns - self.baseline(states).detach()

	def _format_paths(self, **paths):

		paths = super()._format_paths(**paths)

		if 'advantages' not in paths:
			if 'advantages' not in paths:
				paths['advantages'] = self.compute_advantages(**paths)

		return paths


class Normalizable_Advantage_Agent(Advantage_Agent):

	def __init__(self, normalize_adv=False, **other):
		super().__init__(**other)
		self.normalize_adv = normalize_adv

	def compute_advantages(self, **paths):
		advantages = super().compute_advantages(**paths)

		if self.normalize_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()

		return advantages



