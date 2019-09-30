
import sys, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from ... import framework as fm
from ... import models
from ... import util

from .mixins import *

# PPO using Clipping
class PPOClip(Normalizable_Advantage_Agent,
              Differentiable_Agent,):
	def __init__(self, clip=0.3, **other):
		super().__init__(**other)
		self.clip = clip

		self.stats.new('adv-mean', 'adv-std', 'obj-mean', 'obj-std')

	def gen_action(self, x):
		pi = self.policy.get_pi(x)
		action = pi.sample()
		log_prob = pi.log_prob(action)

		return action, {'log_probs': log_prob}

	def _update_policy(self, states, actions, advantages, log_probs=None, **info):

		if log_probs is None:
			with torch.no_grad():
				pi = self.policy.get_pi(states)
				log_probs = pi.log_prob(actions)
		else:
			if not isinstance(log_probs, torch.Tensor):
				log_probs = torch.cat(log_probs)#.detach().view(-1, ACTION_DIM)
			log_probs = log_probs.view(-1, self.action_dim)

		self.scheduler.step()
		self.stats.update('lr', self.scheduler.get_lr()[0])

		self.stats.update('adv-mean', advantages.mean().detach())
		self.stats.update('adv-std', advantages.std().detach())

		dataloader = DataLoader(TensorDataset(states, actions, advantages, log_probs),
		                        batch_size=self.batch_size, shuffle=True, num_workers=0)

		for ppo_iter in range(self.epochs):

			for idx, (S, A, V, LP) in enumerate(
					dataloader):
				self.optim.zero_grad()

				pi = self.policy.get_pi(S)

				ALP = pi.log_prob(A)

				ratio = torch.exp(ALP - LP)

				objective = torch.min(ratio * V,
				                      ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * V)

				self.stats.update('obj-mean', objective.mean().detach())
				self.stats.update('obj-std', objective.std().detach())

				policy_net_loss = -objective
				policy_net_loss = policy_net_loss.mean()
				policy_net_loss.backward()

				self.optim.step()