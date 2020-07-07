
import sys, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from foundation import framework as fm
from foundation import util

class StochasticPolicy(fm.Policy):
	def get_action(self, state, greedy=False):
		greedy = not self.training if greedy is None else greedy

		pi = self.get_pi(state)
		return util.MLE(pi) if greedy else pi.sample()

	def get_pi(self, obs, include_old=False):
		raise NotImplementedError



class NormalPolicy(StochasticPolicy):
	def __init__(self, model):
		super().__init__(model.din, model.dout//2)
		self.model = model

	def get_pi(self, x, include_old=False):
		mu, log_sigma = self(x)
		sigma = log_sigma.exp()
		pi = Normal(mu, sigma)
		if include_old:
			return pi, Normal(mu.detach(), sigma.detach())
		return pi

	def forward(self, x):
		y = self.model(x.view(-1,self.model.din))
		dim = y.size(-1)//2
		return y.narrow(-1, 0, dim), y.narrow(-1, dim, dim)