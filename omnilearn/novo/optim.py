import torch
from torch import optim as O
from omnibelt import agnostic, unspecified_argument

from omniplex import hparam, inherit_hparams, get_builder#, module, Submodule, with_hparams
from omniplex import Prepared, RegistryBuilder, RegisteredProduct, Structured

from . import base as reg


# class AbstractOptimizer(Prepared):
# 	def step(self, info):
# 		raise NotImplementedError
#
#
# 	def _prepare(self, parameters=None, **kwargs):
# 		return super()._prepare(**kwargs)
#
#
#
# class OptimizerBuilder(reg.BranchBuilder, branch='optim'):
# 	@agnostic
# 	def build(self, ident, parameters=None, **kwargs):
# 		optim = super().build(ident=ident, **kwargs)
# 		if parameters is not None:
# 			optim.prepare(parameters=parameters)
# 		return optim
#
#
#
# class PytorchOptimizer(reg.Product, Structured, AbstractOptimizer, O.Optimizer, registry=OptimizerBuilder):
# 	def __init__(self, params=None, **kwargs):
# 		kwargs = self._extract_hparams(kwargs)
# 		hparams = {k: getattr(self, k) for k, v in self.named_hyperparameters() if v.in_init}
# 		kwargs.update(hparams)
# 		if params is None:
# 			params = [torch.zeros(0)]
# 		super().__init__(params=params, **kwargs)
# 		self.param_groups.clear()
#
#
# 	class Hyperparameter(Structured.Hyperparameter):
# 		def __init__(self, name=None, in_init=True, **kwargs):
# 			super().__init__(name=name, **kwargs)
# 			self.in_init = in_init
#
#
# 	_loss_key = 'loss'
#
# 	def _prepare(self, parameters=None, **kwargs):
# 		if parameters is not None:
# 			self.add_parameters(*parameters)
# 		return super()._prepare(parameters=parameters, **kwargs)
#
#
# 	def add_parameters(self, *parameters):
# 		param_groups = list(parameters)
# 		if len(param_groups) == 0:
# 			raise ValueError("optimizer got an empty parameter list")
# 		if not isinstance(param_groups[0], dict):
# 			param_groups = [{'params': param_groups}]
#
# 		for param_group in param_groups:
# 			self.add_param_group(param_group)
#
#
# 	def _pytorch_step(self):
# 		return super(AbstractOptimizer, self).step()
#
#
# 	def step(self, info):
# 		loss: torch.FloatTensor = info.get(self._loss_key)
# 		if loss is not None:
# 			self.zero_grad()
# 			loss.backward()
# 			self._pytorch_step()
# 		return info
#
#
#
# class SGD(PytorchOptimizer, O.SGD, ident='sgd'):
# 	lr = hparam(required=True)
# 	momentum = hparam(0.)
# 	dampening = hparam(0.)
# 	weight_decay = hparam(0.)
# 	nesterov = hparam(False)
#
#
#
# class ASGD(PytorchOptimizer, O.ASGD, ident='asgd'):
# 	lr = hparam(0.01)
# 	lambd = hparam(0.0001)
# 	alpha = hparam(0.75)
# 	t0 = hparam(1000000.0)
# 	weight_decay = hparam(0.)
#
#
#
# class Adadelta(PytorchOptimizer, O.Adadelta, ident='adadelta'):
# 	lr = hparam(1.0)
# 	rho = hparam(0.9)
# 	eps = hparam(1e-06)
# 	weight_decay = hparam(0.)
#
#
#
# class Adagrad(PytorchOptimizer, O.Adagrad, ident='adagrad'):
# 	lr = hparam(0.01)
# 	lr_decay = hparam(0)
# 	weight_decay = hparam(0.)
# 	initial_accumulator_value = hparam(0.)
# 	eps = hparam(1e-10)
#
#
#
# class AdamLike(PytorchOptimizer):
# 	lr = hparam(0.001)
# 	beta1 = hparam(0.9, in_init=False)
# 	beta2 = hparam(0.999, in_init=False)
# 	eps = hparam(1e-8)
# 	weight_decay = hparam(0.)
#
# 	def __init__(self, beta1=None, beta2=None, **kwargs):
# 		if beta1 is None:
# 			beta1 = self.beta1
# 		if beta2 is None:
# 			beta2 = self.beta2
# 		super().__init__(betas=(beta1, beta2), **kwargs)
#
#
# @inherit_hparams('lr', 'beta1', 'beta2', 'eps', 'weight_decay')
# class Adam(AdamLike, O.Adam, ident='adam', is_default=True):
# 	amsgrad = hparam(False)
#
#
#
# @inherit_hparams('lr', 'beta1', 'beta2', 'eps')
# class AdamW(AdamLike, O.AdamW, ident='adamw'):
# 	weight_decay = hparam(0.01)
# 	amsgrad = hparam(False)
#
#
#
# @inherit_hparams('beta1', 'beta2', 'eps', 'weight_decay')
# class Adamax(AdamLike, O.Adamax, ident='adamax'):
# 	lr = hparam(0.002)
#
#
#
# class RMSprop(PytorchOptimizer, O.RMSprop, ident='rmsprop'):
# 	lr = hparam(0.01)
# 	alpha = hparam(0.99)
# 	eps = hparam(1e-8)
# 	weight_decay = hparam(0.)
# 	momentum = hparam(0.)
# 	centered = hparam(False)
#
#
#
# class Rprop(PytorchOptimizer, O.Rprop, ident='rprop'):
# 	lr = hparam(0.01)
# 	eta1 = hparam(0.5)
# 	eta2 = hparam(1.2)
# 	step1 = hparam(1e-06)
# 	step2 = hparam(50)







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




