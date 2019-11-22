import torch
from torch.optim import Optimizer
from foundation.foundation import util

def flatten_params(params):
	return torch.cat([param.clone().view(-1) for param in params])

class Stat_Optim:
	def __init__(self, optim):
		self.optim = optim
		
	def zero_grad(self):
		self.optim.zero_grad()
	
	def step(self, closure=None):
		
		self.optim.step(closure)
		
		grad = torch.cat([p.grad.data.view(-1) for p in self.optim.param_groups[0]['params']], 0)  # flat vpg
		stats = util.StatsMeter('grad-norm')
		stats.update('grad-norm', grad.norm().item())
		
		return stats
		

class Conjugate_Gradient(Optimizer):
	def __init__(self, params, step_size, nsteps, residual_tol=1e-10, ret_stats=False):
		super(Conjugate_Gradient, self).__init__(params, {'step-size':step_size,
		                                                  'nsteps': nsteps,
		                                                  'res-tol': residual_tol,
		                                                  'ret_stats':ret_stats})
	
	def cg_solve(self, apply_A, b, x0=None, res_tol=1e-10, nsteps=10):
		
		if x0 is None:
			x = torch.zeros(*b.size())
			r = b.clone()
		else:
			x = x0
			r = b - apply_A(x)
		
		p = r.clone()
		rdotr = r @ r
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
		
		return x
	
	def step(self, apply_A):
		
		for group in self.param_groups:
		
			# get vpg (from backward pass)
			vpg = torch.cat([p.grad.data.view(-1) for p in group['params'] if p.grad is not None], 0)  # flat vpg
			
			# compute npg with cg
			npg = self.cg_solve(apply_A, vpg, res_tol=group['res-tol'], nsteps=group['nsteps'])
			
			# compute learning rate
			alpha = (2 * group['step-size'] / (vpg @ npg)).sqrt()
			
			# update params
			i = 0
			for param in self.param_groups[0]['params']:
				l = param.numel()
				param.data.add_(npg[i:i + l].view(*param.size()).mul_(alpha)) # gradient ascent
				i += l
			
		if group['ret_stats']:
			stats = util.StatsMeter('vpg-norm', 'npg-norm', 'alpha')
			stats.update('alpha', alpha.item())
			stats.update('vpg-norm', vpg.norm().item())
			stats.update('npg-norm', npg.norm().item())
			return stats