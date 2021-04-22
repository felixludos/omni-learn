
import sys, os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as distrib
from torch.distributions.utils import lazy_property

from omnibelt import InitWall, unspecified_argument
import omnifig as fig

from .features import Deviced, DeviceBase

class DistributionBase(DeviceBase, distrib.Distribution):
	def __init__(self, params={}, **kwargs):
		kwargs.update(params)
		super().__init__(**kwargs)
		self._params = list(params.keys())
		self.to(self.device)
	
	def to(self, device):
		for p in self._params:
			data = getattr(self, p, None)
			if data is not None:
				setattr(self, p, data.to(device))

class Distribution(Deviced, fig.Configurable, DistributionBase):
	
	def sample(self, sample_shape=torch.Size()):
		if isinstance(sample_shape, int):
			sample_shape = sample_shape,
		return super().sample(sample_shape)


@fig.Component('distribution/normal')
class Normal(Distribution, distrib.Normal):
	def __init__(self, A, loc=unspecified_argument, scale=unspecified_argument,
	             logscale=unspecified_argument, params={}, **kwargs):
		if loc is unspecified_argument:
			loc = A.pull('loc', 0.)
		loc = torch.as_tensor(loc).float()
		if scale is unspecified_argument:
			scale = A.pull('scale', None)
		if scale is None:
			if logscale is unspecified_argument:
				logscale = A.pull('logscale', 0.)
			logscale = torch.as_tensor(logscale).float()
			scale = logscale.exp()
		else:
			scale = torch.as_tensor(scale).float()
		
		params.update({'loc':loc, 'scale':scale})
		super().__init__(A, params=params, **kwargs)


@fig.Component('distribution/categorical')
class Categorical(Distribution, distrib.Categorical):
	def __init__(self, A, probs=unspecified_argument, logits=unspecified_argument, params={}, **kwargs):
		if probs is unspecified_argument:
			probs = A.pull('probs', None)
		if probs is not None:
			probs = torch.as_tensor(probs)
		if logits is unspecified_argument:
			logits = A.pull('logits', None)
			if logits is None:
				num = A.pull('num-cats', 2)
				logits = [0.] * num
		if logits is not None:
			logits = torch.as_tensor(logits)

		params.update({'probs':probs, 'logits':logits})
		super().__init__(A, params=params, **kwargs)
		

@fig.Component('distribution/vonmises')
class VonMises(Distribution, distrib.VonMises):
	def __init__(self, A, loc=unspecified_argument, concentration=unspecified_argument,
	             logconc=unspecified_argument, params={}, **kwargs):
		if loc is unspecified_argument:
			loc = A.pull('loc', 0.)
		if loc is not None:
			loc = torch.as_tensor(loc)
		if concentration is unspecified_argument:
			concentration = A.pull('scale', None)
		if concentration is None:
			if logconc is unspecified_argument:
				logconc = A.pull('logconc', 0.)
			logconc = torch.as_tensor(logconc).float()
			concentration = logconc.exp()
		else:
			concentration = torch.as_tensor(concentration).float()
		
		params.update({'loc':loc, 'concentration':concentration})
		super().__init__(A, params=params, **kwargs)
		
