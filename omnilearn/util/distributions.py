
import sys, os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as distrib
from torch.distributions.utils import lazy_property
from torch.distributions import constraints

from omnibelt import InitWall, unspecified_argument
import omnifig as fig

from .features import Deviced, DeviceBase

class Distribution(DeviceBase, distrib.Distribution):
	def __init__(self, *args, device=None, **kwargs):
		if device is not None:
			args = tuple((a.to(device) if isinstance(a, torch.Tensor) else a) for a in args)
			kwargs = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in kwargs.items()}
		super().__init__(*args, device=device, **kwargs)
		# self.to(self.device)
	
	def sample(self, sample_shape=torch.Size()):
		if isinstance(sample_shape, int):
			sample_shape = sample_shape,
		return super().sample(sample_shape)
	
	def bsample(self):
		'''
		Get the best sample (by default, the mean, but possibly also the mode)
		:return:
		'''
		return self.mean
	
	def to(self, device):
		raise NotImplementedError # TODO
		for p in self.arg_constraints.keys():
			data = getattr(self, p, None)
			if data is not None:
				setattr(self, p, data.to(device))


class DistributionFunction(Deviced, fig.Configurable, Distribution):
	pass


_available_base_distributions = {}
for x in distrib.__dict__['__all__']:
	t = getattr(distrib, x)
	if type(t) == type and t.__name__ !='Distribution' and issubclass(t, distrib.Distribution):
		_available_base_distributions[x.lower()] = type(x, (Distribution, t), {})
locals().update({t.__name__:t for t in _available_base_distributions.values()})

# class NormalBase(DistributionBase, distrib.Normal):
# 	pass
# _available_base_distributions['normal'] = NormalBase

def constrain_real(cons, real, eps=1e-12):
	
	if cons == constraints.simplex:
		return F.normalize(real, p=1, dim=1).abs()
	
	if isinstance(cons, (constraints.greater_than, constraints.greater_than_eq)) and cons.lower_bound == 0:
		return real.exp()
	
	if isinstance(cons, constraints.interval):
		mn, mx = cons.lower_bound, cons.upper_bound
		assert mn == 0 and mx == 1, f'{cons}: {mn} {mx}'
		return real.sigmoid()
	
	if cons == constraints.positive_definite:
		raise NotImplementedError # TODO
		if len(real.shape) == 1:
			N = int(np.sqrt(len(real)))
			real = real.reshape(N,N) + eps*torch.eye(N, device=real.device)
		return real.t() @ real
	
	return real

def get_distrib_param_size(distrib_cls, key, dim):
	cons = distrib_cls.arg_constraints[key]
	if cons == constraints.positive_definite:
		return dim**2
	return dim


@fig.AutoComponent('distribution-base')
def get_distribution_base(ident):
	if not isinstance(ident, str):
		return ident
	return _available_base_distributions.get(ident.lower())


class TorchDistribution(DistributionFunction):
	def __init__(self, A, **kwargs):
		params = self._config_params(A, **kwargs)
		super().__init__(A, **params, **kwargs)

	@classmethod
	def _config_params(cls, A, trust_params=None, **kwargs):

		if trust_params is None:
			trust_params = A.pull('trust-params', False)

		constraints = cls.arg_constraints.copy()

		if 'probs' in constraints and 'logits' in constraints and kwargs.get('use_logits', True) and \
				('use_logits' in kwargs or A.pull('use-logits', True)):
			del constraints['probs']
		else:
			del constraints['logits']

		params = {}
		for key, cons in constraints.items():
			val = None
			if key in kwargs:
				val = kwargs[key]
				del kwargs[key]
			if val is None:
				val = A.pull(key, 0.)
			val = torch.as_tensor(val)
			if not trust_params:
				val = constrain_real(cons, val)
			params[key] = val

		return params


_available_distributions = {k: type(f'{t.__name__}Function', (TorchDistribution, t), {})
                                 for k,t in _available_base_distributions.items()}
locals().update({t.__name__:t for t in _available_distributions.values()})

fig.Component('distribution/normal')(_available_distributions['normal'])
fig.Component('distribution/categorical')(_available_distributions['categorical'])
fig.Component('distribution/vonmises')(_available_distributions['vonmises'])

@fig.AutoComponent('torch-distribution')
def get_torch_distribution(ident):
	if not isinstance(ident, str):
		return ident
	return _available_distributions.get(ident.lower())

# class Normal(Distribution, _available_base_distributions['normal']):
# 	pass
# @fig.Component('distribution/categorical')
# class Categorical(Distribution, _available_base_distributions['categorical']):
# 	pass
#
# @fig.Component('distribution/vonmises')
# class VonMises(Distribution, _available_base_distributions['vonmises']):
# 	pass

# TODO automatically register all distributions as components







