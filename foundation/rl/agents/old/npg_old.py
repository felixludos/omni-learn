import torch
import torch.distributions as distrib
from torch.autograd import grad
from foundation.foundation import framework as fm
from foundation.foundation.models import networks as nets
from foundation.foundation import util
from .optim import Conjugate_Gradient, flatten_params
import copy

class NPG(fm.Agent):
	def __init__(self, policy, baseline=None, discount=0.99, step_size=0.01, max_cg_iter=10, reg_coeff=1e-5):
		super(NPG, self).__init__(policy, baseline, discount)
		
		self.reg_coeff = reg_coeff
		
		self.optim = Conjugate_Gradient(self.policy.parameters(), step_size=step_size, nsteps=max_cg_iter, ret_stats=True)
		
		self.running_score = None
		
	def FVP(self, v, grad_kl):
		Fv = flatten_params(grad(grad_kl.clone() @ v, self.policy.parameters(), retain_graph=True))
		
		return Fv + self.reg_coeff * v
	
	def _update_policy(self, paths): # using fully collated paths
		
		stats = util.StatsMeter('objective-delta', 'mean-kl', 'score', 'returns')
		
		stats.update('returns', paths.returns.mean().item())
		
		# evaluate objective
		pi, pi_old = self.policy.get_pi(paths.observations, include_old=True)
		
		eta = self.objective(pi, pi_old, paths.actions, paths.advantages)
		
		self.optim.zero_grad()

		#grads = grad(eta, self.policy.parameters(), retain_graph=True)

		
		eta.backward(retain_graph=True)

		
		grad_kl = flatten_params(
			grad(distrib.kl_divergence(pi, pi_old).mean(),
			     self.policy.parameters(), create_graph=True))
		
		optim_stats = self.optim.step(lambda v: self.FVP(v, grad_kl))
		if optim_stats is not None:
			stats.join(optim_stats, prefix='optim-')
		
		pis = self.policy.get_pi(paths.observations, include_old=True)
		new_eta = self.objective(*pis, actions=paths.actions, advantages=paths.advantages)
		stats.update('mean-kl', distrib.kl_divergence(*pis).mean().item())
		stats.update('objective-delta', new_eta.item() - eta.item())
		
		self.policy.update_old_model()
		
		self.running_score = stats['returns'].avg if self.running_score is None \
			else stats['returns'].avg * 0.1 + self.running_score * 0.9
		stats.update('score', self.running_score)
		
		return stats
		
	def objective(self, pi, pi_old, actions, advantages):
		advantages = advantages / (advantages.abs().max() + 1e-8)
		surr = torch.mean( (pi.log_prob(actions) - pi_old.log_prob(actions)).exp() * advantages ) # CPI surrogate
		return surr

