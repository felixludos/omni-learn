
import sys, os, time
import numpy as np
import torch
from torch.autograd import grad
from torch.distributions import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from ... import framework as fm
from ... import models
from ... import util

from .mixins import *

# Natural Policy Gradients
class NPG(Normalizable_Advantage_Agent,
              Differentiable_Agent,):
	def __init__(self, policy, step_size=0.01, max_cg_iter=10,
	             reg_coeff=1e-5, res_tol=1e-10, **other):

		optim = util.get_optimizer('cg', policy.parameters(),
		                           step_size=step_size, max_cg_iter=max_cg_iter, res_tol=res_tol)

		super().__init__(policy=policy, optim=optim, **other)
		self.reg_coeff = reg_coeff

		self.stats.new('mean-kl', 'adv-mean', 'adv-std', 'obj-mean', 'obj-std')

	def FVP(self, v, grad_kl):
		Fv = nn.utils.parameters_to_vector(grad(grad_kl.clone() @ v, self.policy.parameters(),
		                                        retain_graph=True))
		return Fv + self.reg_coeff * v

	def objective(self, pi, pi_old, actions, advantages):
		advantages = advantages / (advantages.abs().max() + 1e-8)
		return (pi.log_prob(actions) - pi_old.log_prob(actions)).exp() * advantages.view(-1, 1) # CPI surrogate

	def _update_policy(self, states, actions, advantages, **info):

		if self.scheduler is not None:
			self.scheduler.step()
			self.stats.update('lr', self.scheduler.get_lr()[0])

		self.stats.update('adv-mean', advantages.mean().detach())
		self.stats.update('adv-std', advantages.std().detach())

		pi, pi_old = self.policy.get_pi(states, include_old=True)

		params = nn.utils.parameters_to_vector(self.policy.parameters())

		objective = self.objective(pi, pi_old, actions, advantages)
		eta = objective.mean()

		self.stats.update('obj-mean', eta.detach())
		self.stats.update('obj-std', objective.std().detach())

		self.optim.zero_grad()
		eta.backward(retain_graph=True)

		grad_kl = grad(kl_divergence(pi, pi_old).mean(),
		               params, create_graph=True, retain_graph=True)

		self.optim.step(lambda v: self.FVP(v, grad_kl))