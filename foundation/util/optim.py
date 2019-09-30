
import torch
from torch import optim as O
from torch import nn
from torch.optim import Optimizer
from .stats import StatsMeter

def get_optimizer(optim_type, parameters, lr=1e-3, weight_decay=0, momentum=0, **optim_args):
	if optim_type == 'sgd':
		optimizer = O.SGD(parameters,lr=lr, weight_decay=weight_decay, momentum=momentum, **optim_args)
	elif optim_type == 'rmsprop':
		optimizer = O.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, **optim_args)
	elif optim_type == 'adam':
		optimizer = O.Adam(parameters, lr=lr, weight_decay=weight_decay)
	elif optim_type == 'cg':
		optimizer = Conjugate_Gradient(parameters, **optim_args)
	elif optim_type == 'rprop':
		optimizer = O.Rprop(parameters, lr=lr)
	elif optim_type == 'adagrad':
		optimizer = O.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
	elif optim_type == 'adadelta':
		optimizer = O.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
	else:
		assert False, "Unknown optimizer type: " + optim_type
	return optimizer


class Conjugate_Gradient(Optimizer):
	def __init__(self, params, step_size, nsteps, residual_tol=1e-10, ret_stats=False):
		super().__init__(params, {'step-size': step_size,
                                  'nsteps': nsteps,
                                  'res-tol': residual_tol,
                                  'ret_stats': ret_stats})

		self.stats = [StatsMeter('v-norm', 'n-norm', 'alpha') for _ in self.param_groups]

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

		for group, stats in zip(self.param_groups, self.stats):

			# get v (from backward pass)
			p = nn.parameters_to_vector(group)
			v = p.grad

			# compute n with cg
			n = self.cg_solve(apply_A, v, res_tol=group['res-tol'], nsteps=group['nsteps'])

			# compute learning rate
			alpha = (2 * group['step-size'] / (v @ n)).sqrt()

			p.data.add( n.mul_(alpha) )
			nn.vector_to_parameters(p, group)

			if group['ret_stats']:
				stats.update('alpha', alpha)
				stats.update('v-norm', v.norm())
				stats.update('n-norm', n.norm())