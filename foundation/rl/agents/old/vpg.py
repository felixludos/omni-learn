

import torch
import torch.distributions as distrib
from torch.autograd import grad
from foundation.foundation import framework as fm
from foundation.foundation.models import networks as nets
from foundation.foundation import util
from .optim import Stat_Optim
#import matplotlib.pyplot as plt
#plt.switch_backend('TkAgg')


class VPG(fm.Agent):
	def __init__(self, policy, baseline=None, discount=0.99, optim_type='sgd', lr=1e-2, weight_decay=1e-4, momentum=0):
		super(VPG, self).__init__(policy, baseline, discount)
		
		self.optim = nets.get_optimizer(optim_type, self.policy.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
		self.optim = Stat_Optim(self.optim)
		
		self.running_score = None
	
	def train_step(self, paths):
		
		stats = util.StatsMeter('returns', 'score', 'objective')
			
		paths.returns = self.compute_returns(paths.rewards)
		paths.advantages = self.compute_advantages(paths.returns, paths.observations)

		baseline_stats = self.baseline.train_step(paths)
		if baseline_stats is not None:
			stats.join(baseline_stats, prefix='bsln-')

		paths.returns = torch.cat(paths.returns)
		paths.advantages = torch.cat(paths.advantages)
		paths.observations = torch.cat(paths.observations)
		paths.actions = torch.cat(paths.actions)

		stats.update('returns', paths.returns.mean().item())

		# evaluate objective
		pi = self.policy.get_pi(paths.observations)
		eta = self.objective(pi, paths.actions, paths.advantages)
		stats.update('objective', eta.item())

		self.optim.zero_grad()
		eta.mul(-1).backward() # maximize objective
		optim_stats = self.optim.step()
		if optim_stats is not None:
			stats.join(optim_stats, prefix='optim-')
		
		self.running_score = stats['returns'].avg if self.running_score is None \
			else stats['returns'].avg * 0.1 + self.running_score * 0.9
		stats.update('score', self.running_score)
		
		return stats
	
	def objective(self, pi, actions, advantages):
		advantages = advantages / (advantages.abs().max() + 1e-8)
		return (pi.log_prob(actions) * advantages).mean()















