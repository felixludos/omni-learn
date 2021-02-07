
from itertools import chain
import torch
from torch import optim as O
from torch import nn
from torch.optim import Optimizer as PytorchOptimizer

from omnibelt import LoadedValue, get_printer

prt = get_printer(__name__)

from .stats import StatsMeter

import omnifig as fig

# region Optimizers

class OptimizerBase(PytorchOptimizer):
	def __init__(self, A=None, **settings):
		super().__init__(params=[torch.zeros(0)], **settings)
		self.param_groups.clear()
	
	def __setstate__(self, state):
		groups = state.get('param_groups', None)
		if groups is not None:
			for group, new in zip(self.param_groups, groups):
				group.update({k:v for k,v in new.items() if not isinstance(v, LoadedValue)})
			del state['param_groups']
		super().__setstate__(state)
	
	def prep(self, params):
		
		param_groups = list(params)
		if len(param_groups) == 0:
			raise ValueError("optimizer got an empty parameter list")
		if not isinstance(param_groups[0], dict):
			param_groups = [{'params': param_groups}]
		
		for param_group in param_groups:
			self.add_param_group(param_group)


@fig.AutoComponent('sgd', auto_name=False)
class SGD(OptimizerBase, O.SGD):
	def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
		super().__init__(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

@fig.AutoComponent('asgd', auto_name=False)
class ASGD(OptimizerBase, O.ASGD):
	def __init__(self, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
		super().__init__(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)


@fig.AutoComponent('adadelta', auto_name=False)
class Adagrad(OptimizerBase, O.Adadelta):
	def __init__(self, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
		super().__init__(lr=lr, rho=rho, weight_decay=weight_decay, eps=eps)
		
@fig.AutoComponent('adagrad', auto_name=False)
class Adagrad(OptimizerBase, O.Adagrad):
	def __init__(self, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
		super().__init__(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
		                 initial_accumulator_value=initial_accumulator_value, eps=eps)


@fig.AutoComponent('adam', auto_name=False)
class Adam(OptimizerBase, O.Adam):
	def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, amsgrad=False):
		super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

@fig.AutoComponent('adamw', auto_name=False)
class AdamW(OptimizerBase, O.AdamW):
	def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2, amsgrad=False):
		super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

@fig.AutoComponent('adamax', auto_name=False)
class Adamax(OptimizerBase, O.Adamax):
	def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0):
		super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)


@fig.AutoComponent('rmsprop', auto_name=False)
class RMSprop(OptimizerBase, O.RMSprop):
	def __init__(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
		super().__init__(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)

@fig.AutoComponent('rprop', auto_name=False)
class Rprop(OptimizerBase, O.Rprop):
	def __init__(self, lr=0.01, eta1=0.5, eta2=1.2, step1=1e-06, step2=50):
		super().__init__(lr=lr, etas=(eta1, eta2), step_sizes=(step1, step2))


try:
	from ranger import Ranger as RangerOptim
	
	@fig.AutoComponent('ranger', auto_name=False)
	class Ranger(OptimizerBase, RangerOptim):
		def __init__(self, lr=0.001, alpha=0.5, k=6, N_sma_threshhold=5, beta1=.95, beta2=0.999, eps=1e-5, weight_decay=0):
			super().__init__(lr=lr, alpha=alpha, k=k, N_sma_threshhold=N_sma_threshhold, betas=(beta1,beta2),
			                 eps=eps, weight_decay=weight_decay)

except ImportError:
	prt.info('failed to import Ranger optimizer')



# endregion


# @fig.AutoModifier('schedulable')
# class Schedulable(BaseOptimizer):
# 	def __init__(self, A):
# 		scheduler = A.pull('scheduler', None)
# 		super().__init__(A)
#
# 		self.scheduler = scheduler
#
# 	def prep(self, params):
# 		super().prep(params)
# 		if self.scheduler is not None:
# 			self.scheduler.prep(self)
#
# 	def __repr__(self):
# 		base = super().__repr__()
# 		if self.scheduler is not None:
# 			title, *rest = base.split('\n')
# 			sch = repr(self.scheduler)
# 			base = '\n'.join((title, sch, *rest))
# 		return base
#
# 	def load_state_dict(self, state_dict, strict=True):
# 		if self.scheduler is not None and 'scheduler' in state_dict:
# 			self.scheduler.load_state_dict(state_dict['scheduler'])
# 		super().load_state_dict(state_dict, strict=strict)
#
# 	def state_dict(self, *args, **kwargs):
# 		state_dict = super().state_dict(*args, **kwargs)
# 		if self.scheduler is not None:
# 			state_dict['scheduler'] = self.scheduler.state_dict()
# 		return state_dict


# region Complex extensions


class Complex_Optimizer(OptimizerBase):
	def __init__(self, **optims):
		self.__dict__['optims'] = None
		super().__init__(defaults=None)
		self.__dict__['optims'] = optims
		self.group_params()
		# self._update_groups()
		# self._update_named_groups()
		# self.__dict__['param_groups'] = [grp for grp in chain(*[o.param_groups for o in optims.values()])]

	def group_params(self):
		self.param_groups = sum([[p for p in o.param_groups] for o in self.optims.values()], [])

	def _update_groups(self):
		self.param_groups = self._param_group_gen()

	def _param_group_gen(self):
		for optim in self.optims.values():
			for grp in optim.param_groups:
				yield grp
		self._update_groups()

	def _update_named_groups(self):
		self.named_param_groups = self._named_param_group_gen()

	def _named_param_group_gen(self):
		for key, optim in self.optims.items():
			for grp in optim.param_groups:
				yield (key, grp)
		self._update_named_groups()

	def add_param_group(self, *args, **kwargs): # probably shouldnt be used
		return
		# raise Exception('invalid for complex optimizers')

	def load_state_dict(self, state_dict):
		for name, optim in self.optims.items():
			optim.load_state_dict(state_dict[name])
		self.group_params() # important to get loaded param groups

	def state_dict(self):
		return {name:optim.state_dict() for name, optim in self.optims.items()}

	def zero_grad(self):
		for optim in self.optims.values():
			optim.zero_grad()

	def step(self, closure=None): # can and perhaps should be overridden
		for optim in self.optims.values():
			optim.step(closure)

	def __iter__(self):
		return iter(self.optims)
	def items(self):
		return self.optims.items()

	def __len__(self):
		return len(self.optims)

	def __getitem__(self, item):
		return self.optims[item]
	def __setitem__(self, key, value):
		# raise NotImplementedError
		self.optims[key] = value
	def __delitem__(self, key):
		del self.optims[key]

	def __getattr__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return self.optims[item]
	def __setattr__(self, key, value):
		# raise NotImplementedError
		if isinstance(value, OptimizerBase):
			# raise NotImplementedError
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
			s.append('  {}: {}'.format(k,str(v).replace('\n', '\n    ')))

		s.append(')')

		return '\n'.join(s)


# endregion

## region old


# def get_optimizer(optim_type, parameters, lr=1e-3, weight_decay=0, momentum=0, beta1=.9, beta2=.999, **optim_args):
# 	if optim_type == 'sgd':
# 		optimizer = O.SGD(parameters,lr=lr, weight_decay=weight_decay, momentum=momentum, **optim_args)
# 	elif optim_type == 'rmsprop':
# 		optimizer = O.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, **optim_args)
# 	elif optim_type == 'adam':
# 		optimizer = O.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), **optim_args)
# 	elif optim_type == 'adamw':
# 		optimizer = O.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), **optim_args)
# 	elif optim_type == 'ranger':
# 		optimizer = Ranger(parameters, lr=lr, weight_decay=weight_decay,betas=(beta1, beta2), **optim_args) #alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999))
# 	elif optim_type == 'cg':
# 		optimizer = Conjugate_Gradient(parameters, **optim_args)
# 	elif optim_type == 'rprop':
# 		optimizer = O.Rprop(parameters, lr=lr)
# 	elif optim_type == 'adagrad':
# 		optimizer = O.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
# 	elif optim_type == 'adadelta':
# 		optimizer = O.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
# 	else:
# 		assert False, "Unknown optimizer type: " + optim_type
# 	return optimizer

# def default_create_optim(parameters, info): # info is a train.Config
#
# 	kwargs = {
#
# 		'optim_type': info.pull('optim_type'),
#
# 		'parameters': parameters,
#
# 		'lr': info.pull('lr', 1e-3),
# 		'weight_decay': info.pull('weight_decay', 0),
# 		'momentum': info.pull('momentum', 0),
#
# 		'beta1': info.pull('beta1', .9),
# 		'beta2': info.pull('beta2', .999),
# 	}
#
# 	return get_optimizer(**kwargs)
#
# def default_create_scheduler(optimizer, info):
#
# 	if 'scheduler_type' not in info:
# 		return None
#
# 	name = info.pull('scheduler_type')
#
# 	freq = info.pull('scheduler_freq', None)
# 	# if freq is None:
# 	# 	print('Scheduler will step at the end of each epoch')
# 	# else:
# 	# 	print(f'Scheduler will step every {freq} iterations')
# 	cut_after = info.pull('scheduler_cut_after', None)
#
# 	min_lr = info.pull('scheduler_min_lr', 0.)
# 	last = info.pull('scheduler_last_epoch', -1)
#
# 	common = {
# 		'cut_after': cut_after,
# 		'freq': freq,
# 	}
#
# 	if name == 'step':
#
# 		factor = info.pull('scheduler_decay', 0.1)
#
# 		step_size = info.pull('scheduler_step')
# 		out = StepLR(optimizer, step_size=step_size, gamma=factor, last_epoch=last,
# 		             req_loss=False, **common)
# 	elif name == 'plateau':
#
# 		factor = info.pull('scheduler_decay', 0.1)
# 		patience = info.pull('scheduler_patience', 10)
# 		cooldown = info.pull('scheduler_cooldown', 0)
#
# 		out = ReduceOnPlateau(optimizer, factor=factor, patience=patience, verbose=True,
# 		                                       min_lr=min_lr, cooldown=cooldown,
# 		                      req_loss=True, **common)
#
# 	elif name == 'lambda':
#
# 		func = info.pull('scheduler_lambda')
#
# 		out = LambdaLR(optimizer, lr_lambda=func, last_epoch=last, **common)
#
# 	elif name == 'cos':
# 		num_steps = info.pull('scheduler_total_steps', None) # num_steps
# 		if num_steps is None:
# 			if freq is None or freq <= 0:
# 				raise Exception('cos scheduler needs to know the max number of steps')
# 			num_steps = info.pull('scheduler_total_iterations', '<>training.step_limit') // freq \
# 				- info.pull('scheduler_early_stop', 0)
#
# 		eta_min = min_lr
#
# 		out = CosineAnnealing(optimizer, T_max=num_steps, eta_min=eta_min, last_epoch=last,
# 		                      req_loss=False, **common)
#
# 	else:
# 		raise Exception(f'unknown name {name}')
#
# 	return out

# region old

# class Base_Scheduler(O.lr_scheduler._LRScheduler):
#
# 	def __init__(self, *args, cut_after=None, freq=None, req_loss=None, **kwargs):
# 		if freq is not None and freq <= 0:
# 			freq = None
# 		self.freq = freq
# 		self.cut_after = cut_after
# 		self.req_loss = req_loss
# 		super().__init__(*args, **kwargs)
#
# 	def requires_loss(self):
# 		return self.req_loss
#
# 	def epoch_end(self, *args, **kwargs):
# 		if self.freq is None:
# 			self.step(*args, **kwargs)
#
# 	def maintain(self, step, *args, **kwargs):
# 		if self.freq is not None and step % self.freq == 0:
# 			self.step(*args, **kwargs)
#
# 	def step(self, *args, **kwargs):
# 		if self.cut_after is not None and self._step_count > self.cut_after:
# 			return
# 		return super().step(*args, **kwargs)
#
# class StepLR(Base_Scheduler, O.lr_scheduler.StepLR):
# 	def __str__(self):
# 		return self.__repr__()
#
# 	def __repr__(self):
# 		return 'StepLR(step={}, gamma={})'.format(self.step_size, self.gamma)
#
# class LambdaLR(Base_Scheduler, O.lr_scheduler.LambdaLR):
#
# 	def __str__(self):
# 		return self.__repr__()
#
# 	def __repr__(self):
# 		return f'LambdaLR(lambda={self.lr_lambdas[0]})'
#
# class MultiStepLR(Base_Scheduler, O.lr_scheduler.MultiStepLR):
# 	def __str__(self):
# 		return self.__repr__()
#
# 	def __repr__(self):
# 		return 'MultiStepLR(milestones={}, gamma={})'.format(self.milestones, self.gamma)
#
# class ReduceOnPlateau(Base_Scheduler, O.lr_scheduler.ReduceLROnPlateau):
# 	def __str__(self):
# 		return self.__repr__()
#
# 	def __repr__(self):
# 		return 'ReduceOnPlateau(factor={}, patience={}, cooldown={})'.format(self.factor, self.patience, self.cooldown)
#
# class CosineAnnealing(Base_Scheduler, O.lr_scheduler.CosineAnnealingLR):
# 	def __str__(self):
# 		return self.__repr__()
#
# 	def __repr__(self):
# 		return 'CosineAnnealing(T_max={},eta_min={})'.format(self.T_max, self.eta_min)

# endregion


# class Conjugate_Gradient(Optimizer):
# 	def __init__(self, params, step_size, nsteps, residual_tol=1e-10, ret_stats=False):
# 		super().__init__(params, {'step-size': step_size,
#                                   'nsteps': nsteps,
#                                   'res-tol': residual_tol,
#                                   'ret_stats': ret_stats})
#
# 		self.stats = [StatsMeter('v-norm', 'n-norm', 'alpha') for _ in self.param_groups]
#
# 	def cg_solve(self, apply_A, b, x0=None, res_tol=1e-10, nsteps=10):
#
# 		if x0 is None:
# 			x = torch.zeros(*b.size())
# 			r = b.clone()
# 		else:
# 			x = x0
# 			r = b - apply_A(x)
#
# 		p = r.clone()
# 		rdotr = r @ r
# 		for i in range(nsteps):
# 			Ap = apply_A(p)
# 			alpha = rdotr / (p @ Ap)
# 			x += alpha * p
# 			r -= alpha * Ap
# 			new_rdotr = r @ r
# 			beta = new_rdotr / rdotr
# 			p = r + beta * p
# 			rdotr = new_rdotr
# 			if rdotr < res_tol:
# 				break
#
# 		return x
#
# 	def step(self, apply_A):
#
# 		for group, stats in zip(self.param_groups, self.stats):
#
# 			# get v (from backward pass)
# 			p = nn.parameters_to_vector(group)
# 			v = p.grad
#
# 			# compute n with cg
# 			n = self.cg_solve(apply_A, v, res_tol=group['res-tol'], nsteps=group['nsteps'])
#
# 			# compute learning rate
# 			alpha = (2 * group['step-size'] / (v @ n)).sqrt()
#
# 			p.data.add( n.mul_(alpha) )
# 			nn.vector_to_parameters(p, group)
#
# 			if group['ret_stats']:
# 				stats.update('alpha', alpha)
# 				stats.update('v-norm', v.norm())
# 				stats.update('n-norm', n.norm())
				
				
# endregion


