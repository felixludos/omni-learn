import torch
from torch import optim as O
from omnibelt import agnosticmethod, unspecified_argument
from omnidata.framework import hparam, inherit_hparams, Parameterized, Builder, register_builder, spaces
from omnidata.framework.building import ClassBuilder, AutoClassBuilder
from omnidata.framework.features import Prepared


class Optimizer(Parameterized, Prepared):
	def __init__(self, **kwargs):
		kwargs = self._extract_hparams(kwargs)
		hparams = {k: getattr(self, k) for k, v in self.named_hyperparameters() if v.in_init}
		kwargs.update(hparams)
		super().__init__(**kwargs)


	class Hyperparameter(Parameterized.Hyperparameter):
		def __init__(self, name=None, in_init=True, **kwargs):
			super().__init__(name=name, **kwargs)
			self.in_init = in_init


	def step(self, info):
		raise NotImplementedError



@register_builder('optim')
@register_builder('optimizer')
class PytorchOptimizer(AutoClassBuilder, Optimizer, O.Optimizer, create_registry=True):
	def __init__(self, params=None, **kwargs):
		if params is None:
			params = [torch.zeros(0)]
		super().__init__(params=params, **kwargs)
		self.param_groups.clear()


	@agnosticmethod
	def _build(self, ident='adam', parameters=None, **kwargs):
		optim = super()._build(ident=ident, **kwargs)
		if parameters is not None:
			optim.prepare(parameters=parameters)
		return optim


	_loss_key = 'loss'


	def _prepare(self, parameters=None, **kwargs):
		if parameters is not None:
			self.add_parameters(*parameters)


	def add_parameters(self, *parameters):
		param_groups = list(parameters)
		if len(param_groups) == 0:
			raise ValueError("optimizer got an empty parameter list")
		if not isinstance(param_groups[0], dict):
			param_groups = [{'params': param_groups}]

		for param_group in param_groups:
			self.add_param_group(param_group)


	def step(self, info):
		loss: torch.FloatTensor = info.get(self._loss_key)
		if loss is not None:
			self.zero_grad()
			loss.backward()
			super(Prepared, self).step()
		return info



# @fig.AutoComponent('sgd', auto_name=False)
class SGD(PytorchOptimizer, O.SGD, ident='sgd'):
	lr = hparam(required=True)
	momentum = hparam(0.)
	dampening = hparam(0.)
	weight_decay = hparam(0.)
	nesterov = hparam(False)

	# def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
	# 	super().__init__(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)


class ASGD(PytorchOptimizer, O.ASGD, ident='asgd'):
	lr = hparam(0.01)
	lambd = hparam(0.0001)
	alpha = hparam(0.75)
	t0 = hparam(1000000.0)
	weight_decay = hparam(0.)

	# def __init__(self, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
	# 	super().__init__(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)


class Adadelta(PytorchOptimizer, O.Adadelta, ident='adadelta'):
	lr = hparam(1.0)
	rho = hparam(0.9)
	eps = hparam(1e-06)
	weight_decay = hparam(0.)

	# def __init__(self, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
	# 	super().__init__(lr=lr, rho=rho, weight_decay=weight_decay, eps=eps)


class Adagrad(PytorchOptimizer, O.Adagrad, ident='adagrad'):
	lr = hparam(0.01)
	lr_decay = hparam(0)
	weight_decay = hparam(0.)
	initial_accumulator_value = hparam(0.)
	eps = hparam(1e-10)

	# def __init__(self, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
	# 	super().__init__(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
	# 	                 initial_accumulator_value=initial_accumulator_value, eps=eps)


class AdamLike(PytorchOptimizer):
	lr = hparam(0.001)
	beta1 = hparam(0.9, in_init=False)
	beta2 = hparam(0.999, in_init=False)
	eps = hparam(1e-8)
	weight_decay = hparam(0.)

	def __init__(self, beta1=None, beta2=None, **kwargs):
		if beta1 is None:
			beta1 = self.beta1
		if beta2 is None:
			beta2 = self.beta2
		super().__init__(betas=(beta1, beta2), **kwargs)


@inherit_hparams('lr', 'beta1', 'beta2', 'eps', 'weight_decay')
class Adam(AdamLike, O.Adam, ident='adam', default=True):
	amsgrad = hparam(False)

	# def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, amsgrad=False):
	# 	super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


@inherit_hparams('lr', 'beta1', 'beta2', 'eps')
class AdamW(AdamLike, O.AdamW, ident='adamw'):
	weight_decay = hparam(0.01)
	amsgrad = hparam(False)

	# def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2, amsgrad=False):
	# 	super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


@inherit_hparams('beta1', 'beta2', 'eps', 'weight_decay')
class Adamax(AdamLike, O.Adamax, ident='adamax'):
	lr = hparam(0.002)

	# def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0):
	# 	super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)


class RMSprop(PytorchOptimizer, O.RMSprop, ident='rmsprop'):
	lr = hparam(0.01)
	alpha = hparam(0.99)
	eps = hparam(1e-8)
	weight_decay = hparam(0.)
	momentum = hparam(0.)
	centered = hparam(False)

	# def __init__(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
	# 	super().__init__(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)


class Rprop(PytorchOptimizer, O.Rprop, ident='rprop'):
	lr = hparam(0.01)
	eta1 = hparam(0.5)
	eta2 = hparam(1.2)
	step1 = hparam(1e-06)
	step2 = hparam(50)

	# def __init__(self, lr=0.01, eta1=0.5, eta2=1.2, step1=1e-06, step2=50):
	# 	super().__init__(lr=lr, etas=(eta1, eta2), step_sizes=(step1, step2))


# try:
# 	from ranger import Ranger as RangerOptim
#
#
# 	class Ranger(OptimizerBase, RangerOptim):
# 		def __init__(self, lr=0.001, alpha=0.5, k=6, N_sma_threshhold=5, beta1=.95, beta2=0.999, eps=1e-5,
# 		             weight_decay=0):
# 			super().__init__(lr=lr, alpha=alpha, k=k, N_sma_threshhold=N_sma_threshhold, betas=(beta1, beta2),
# 			                 eps=eps, weight_decay=weight_decay)
#
# except ImportError:
# 	prt.info('failed to import Ranger optimizer')




# @register_builder('optimizer')
# class BasicOptimizer(ClassBuilder):
# 	ident = hparam('adam', space=['adam'])
#
#
# 	@agnosticmethod
# 	def product_registry(self):
# 		return {
# 			'adam': Adam,
# 			'rmsprop': RMSprop,
# 			'sgd': SGD,
# 			'asgd': ASGD,
# 			'adamw': AdamW,
# 			'adamax': Adamax,
# 			'rprop': Rprop,
# 			'adagrad': Adagrad,
# 			'adadelta': Adadelta,
# 			**super().product_registry()
# 		}
#
#
# 	@agnosticmethod
# 	def _build(self, ident='adam', parameters=None, **kwargs):
# 		optim = super()._build(ident=ident, **kwargs)
# 		if parameters is not None:
# 			optim.prepare(parameters=parameters)
# 		return optim
#
#
# 	class UnknownOptimizer(NotImplementedError):
# 		pass



