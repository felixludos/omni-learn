
from itertools import chain
import torch
from torch import optim as O
from torch import nn
from torch.optim import Optimizer
from .stats import StatsMeter

def get_optimizer(optim_type, parameters, lr=1e-3, weight_decay=0, momentum=0, beta1=.9, beta2=.999, **optim_args):
	if optim_type == 'sgd':
		optimizer = O.SGD(parameters,lr=lr, weight_decay=weight_decay, momentum=momentum, **optim_args)
	elif optim_type == 'rmsprop':
		optimizer = O.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, **optim_args)
	elif optim_type == 'adam':
		optimizer = O.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), **optim_args)
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



class Complex_Optimizer(Optimizer):
	def __init__(self, **optims):
		self.__dict__['optims'] = None
		super().__init__([{}], None)
		self.__dict__['optims'] = optims
		self._update_groups()
		# self.__dict__['param_groups'] = [grp for grp in chain(*[o.param_groups for o in optims.values()])]

	def _update_groups(self):
		self.param_groups = self._param_group_gen()

	def _param_group_gen(self):
		for optim in self.optims.values():
			for grp in optim.param_groups:
				yield grp
		self._update_groups()

	def add_param_group(self, *args, **kwargs): # probably shouldnt be used
		return
		# raise Exception('invalid for complex optimizers')

	def load_state_dict(self, state_dict):
		for name, optim in self.optims.items():
			optim.load_state_dict(state_dict[name])

	def state_dict(self):
		return {name:optim.state_dict() for name, optim in self.optims.items()}

	def zero_grad(self):
		for optim in self.optims.values():
			optim.zero_grad()

	def step(self, closure=None): # can and perhaps should be overridden
		for optim in self.optims.values():
			optim.step(closure)

	def __getitem__(self, item):
		return self.optims[item]
	def __setitem__(self, key, value):
		self.optims[key] = value
	def __delitem__(self, key):
		del self.optims[key]

	def __getattr__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return self.optims[item]
	def __setattr__(self, key, value):
		if isinstance(value, Optimizer):
			self.__setitem__(key, value)
		else:
			super().__setattr__(key, value)
	def __delattr__(self, item):
		try:
			super().__delattr__(item)
		except AttributeError:
			self.__delitem__(item)

	def __str__(self):
		s = ['Complex-Optimizer(']

		for k,v in self.optims.items():
			s.append('  ' + k + ': ' + str(v).replace('\n', '\n    '))

		s.append(')')

		return '\n'.join(s)



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