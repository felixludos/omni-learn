import torch
import torch.distributions as distrib
from torch.autograd import grad
from foundation.foundation import framework as fm
from foundation.foundation.models import networks as nets
from foundation.foundation import util
from .optim import Conjugate_Gradient, flatten_params
import copy


def cg_solve(apply_A, b, x0=None, res_tol=1e-10, nsteps=10):
	if x0 is None:
		x = torch.zeros(*b.size()).type_as(b)
		r = b.clone()
	else:
		x = x0
		r = b - apply_A(x)

	p = r.clone()
	rdotr = r @ r
	# print(res_tol, nsteps)
	# print('rdotr',rdotr)
	for i in range(nsteps):
		Ap = apply_A(p)
		alpha = rdotr / (p @ Ap)
		x += alpha * p
		r -= alpha * Ap
		new_rdotr = r @ r
		beta = new_rdotr / rdotr
		p = r + beta * p
		rdotr = new_rdotr
		if rdotr < res_tol:
			break

	# print('npg', x.norm().item())

	return x

class NPG(fm.Agent):
	def __init__(self, policy, baseline=None, discount=0.99, step_size=0.01, max_cg_iter=10, reg_coeff=1e-5, res_tol=1e-10):
		super(NPG, self).__init__(policy, baseline, discount)
		
		self.reg_coeff = reg_coeff

		self.step_size = step_size
		self.res_tol = res_tol
		self.max_cg_iter = max_cg_iter
		
		self.optim = Conjugate_Gradient(self.policy.parameters(), step_size=step_size, nsteps=max_cg_iter, ret_stats=True)
		
		self.running_score = None
	
	def _update_policy(self, paths): # using fully collated paths
		
		stats = util.StatsMeter('objective-delta', 'mean-kl', 'score', 'returns', 'alpha', 'vpg-norm', 'npg-norm')
		
		stats.update('returns', paths.returns.mean())

		# for k, v in paths.items():
		# 	try:
		# 		print(k, v.size())
		# 	except:
		# 		print(k, len(v))
		#raise Exception('stop')
		
		# evaluate objective
		pi, pi_old = self.policy.get_pi(paths.states, include_old=True)

		#print(paths.observations.size(), paths.actions.size(), paths.advantages.size())
		#print(pi)
		#for p in pi.base:
		#	print(p.logits)

		eta = self.objective(pi, pi_old, paths.actions, paths.advantages)

		#print(eta)

		#print(eta)

		if False and len(pi.base) != 12:
			normal = pi.base[0]
			print(normal.covariance_matrix)

			print(grad(normal.covariance_matrix.sum(), self.policy.model.log_std))
			quit()

			#print(grad(pi.log_prob(pi.sample()).sum(), normal.covariance_matrix))
			print(grad(normal.log_prob(normal.sample()).sum(), normal.covariance_matrix))

			#print(grad(eta, normal.covariance_matrix))
			quit()

		vpg = flatten_params(grad(eta, self.policy.parameters(), retain_graph=True))
		
		grad_kl = flatten_params(
			grad(distrib.kl_divergence(pi, pi_old).mean(),
			     self.policy.parameters(), create_graph=True))

		npg = cg_solve(lambda v: self.FVP(v, grad_kl), vpg, res_tol=self.res_tol, nsteps=self.max_cg_iter)

		alpha = (2 * self.step_size / (vpg @ npg)).sqrt()

		if torch.isnan(alpha).any():
			print('***********problem alpha')
			print('alpha', alpha)
			print('vpg norm',vpg.norm())
			print('npg norm', npg.norm())
			print('grad kl', grad_kl)
			quit()

		i = 0
		for param in self.policy.parameters():
			l = param.numel()
			param.data.add_(npg[i:i + l].view(*param.size()).mul_(alpha))  # gradient ascent
			i += l

		stats.update('alpha', alpha.item())
		stats.update('vpg-norm', vpg.norm().item())
		stats.update('npg-norm', npg.norm().item())

		pis = self.policy.get_pi(paths.states, include_old=True)
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
		surr = torch.mean( (pi.log_prob(actions) - pi_old.log_prob(actions)).exp() * advantages.view(-1, 1) ) # CPI surrogate
		return surr

	def FVP(self, v, grad_kl):
		Fv = flatten_params(grad(grad_kl.clone() @ v, self.policy.parameters(), retain_graph=True))

		return Fv + self.reg_coeff * v

