
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch import distributions as distrib

import omnifig as fig

# import torch
# import torch.nn as nn
# from .. import framework as fm
# from ..op import framework as fm

# from omnibelt import InitWall, unspecified_argument
# import omnifig as fig

# from .layers import make_MLP

from ..op.framework import Function, Generative, Stochastic
from .nets import MultiLayer
from .. import util

###############
# region Prior
###############

class Prior(Generative):
	def __init__(self, A, prior_dim=None, **kwargs):
		super().__init__(A, **kwargs)
		# if prior_dim is None:
		# 	prior_dim = A.pull('latent-dim', '<>din', None, silent=True)
		self.prior_dim = prior_dim

	def generate(self, N=1, prior=None):
		return self._sample(N=1, prior=prior)

	def _sample(self, N, seed=None, prior=None):
		if prior is None:
			prior = self._sample_prior(N, seed=seed)
		return self(prior)

	def sample_prior(self, *shape, seed=None):
		N = int(np.product(shape))
		samples = self._sample_prior(max(N,1), seed=seed)
		shape = *shape, self.prior_dim
		return samples.reshape(*shape)

	def _sample_prior(self, N, seed=None):
		raise NotImplementedError

class SimplePrior(Prior):
	def _sample_prior(self, N, seed=None):
		gen = None
		if seed is not None:
			gen = torch.Generator(self.device)
			gen.manual_seed(seed)
		D = self.prior_dim
		if isinstance(D, int):
			D = (D,)
		return self._simple_prior_sampling(N, *D, device=self.device, generator=gen)
		
	def _simple_prior_sampling(self, *shape, **kwargs):
		raise NotImplementedError

@fig.AutoModifier('gaussian-prior')
class Gaussian(SimplePrior):
	def _simple_prior_sampling(self, *shape, **kwargs):
		return torch.randn(*shape, **kwargs)

@fig.AutoModifier('uniform-prior')
class Uniform(SimplePrior):
	def _simple_prior_sampling(self, *shape, **kwargs):
		return torch.rand(*shape, **kwargs)


@fig.AutoModifier('prior-tfm') # for the generator
class Transformed(Prior):
	def __init__(self, A, **kwargs):

		super().__init__(A, **kwargs)

		prior_tfm = None
		if self.prior_dim is not None:

			tfm_info = A.pull('prior-tfm', None, silent=True, raw=True)
			if tfm_info is not None:
				A.push('prior-tfm.din', self.prior_dim, silent=True)
				A.push('prior-tfm.dout', self.prior_dim, silent=True)

			prior_tfm = A.pull('prior-tfm', None)
		self.prior_tfm = prior_tfm
		if self.prior_tfm is None:
			print('WARNING: not transforming prior')

	def _sample_prior(self, N=1, seed=None):
		q = super()._sample_prior(N, seed=seed)
		return q if self.prior_tfm is None else self.prior_tfm(q)


# endregion


######################
# region Distribution
######################

# TODO: setup mixture model initalization

@fig.AutoModifier('distrib')
class Distribution(Stochastic):
	'''Automodifier that turns a function output into a distribution'''
	def __init__(self, A, autosample=None, distrib_cls=None, joint_params=None, modify_out_layer=None,
	             force_distrib_layer=None, distrib_layer_info=None, constraint_eps=None,
	             **kwargs):
		if autosample is None:
			autosample = A.pull('autosample', None)
		
		if constraint_eps is None:
			constraint_eps = A.pull('constraint-eps', '<>epsilon', 1e-12)
		
		if distrib_cls is None:
			distrib_cls = A.pull('src-distrib')
		distrib_cls = util.get_distribution_base(distrib_cls)
		
		self._distrib_cls = distrib_cls
		constraints = self._distrib_cls.arg_constraints.copy()

		if 'probs' in constraints and 'logits' in constraints:
			if kwargs.get('use_logits', True) and ('use_logits' in kwargs or A.pull('use-logits', True)):
				del constraints['probs']
			else:
				del constraints['logits']
		self._param_constraints = constraints
		self._param_sizes = None
		
		if modify_out_layer is None:
			modify_out_layer = A.pull('modify-out-layer', True)
		if modify_out_layer:
			dout_key = A.pull('_dout_key', None)
			if dout_key is None:
				dout = A.pull('_dout', None)
				if dout is not None:
					dout_key = '_dout'
				else:
					dout = A.pull('dout')
			else:
				dout = A.pull(dout_key)
			chn = dout if isinstance(dout, int) else dout[0]
			
			self._param_sizes = self._compute_distrib_param_sizes(chn)
			total = sum(self._param_sizes)
			
			dout = total if isinstance(dout, int) else (total, *dout[1:])
			if dout_key is None:
				A.push('dout', dout)
			else:
				A.push(dout_key, dout, silent=True)
		
		super().__init__(A, **kwargs)
		
		self.distrib_layer = None
		if not modify_out_layer:
			self.distrib_layer = self._create_distrib_layer(A, distrib_layer_info=distrib_layer_info,
			                                                joint_params=joint_params,
			                                                force_distrib_layer=force_distrib_layer)
		
		self._use_autosample = autosample
		self._constraint_eps = constraint_eps
	
	
	def _create_distrib_layer(self, A, distrib_layer_info=None, joint_params=None, force_distrib_layer=None):
		dout = self.dout
		
		if distrib_layer_info is not None:
			A.push('distrib-layer', distrib_layer_info, silent=True, overwrite=True)
		A.push('distrib-layer._type', 'dense-layer' if isinstance(dout, int) else 'conv-layer',
		       overwrite=False, silent=True)
		A.push('distrib-layer.din', dout, silent=True, overwrite=True)
		layer_info = A.pull('distrib-layer', raw=True)
		
		if joint_params is None:
			joint_params = A.pull('joint-params', True)
		
		if force_distrib_layer is None:
			force_distrib_layer = A.pull('force-distrib-layer', False)

		chn = dout if isinstance(dout, int) else dout[0]
		param_sizes = self._compute_distrib_param_sizes(chn)
		total = sum(param_sizes)
		
		distrib_layer = None
		if joint_params:
			if force_distrib_layer or total != chn:
				pout = total if isinstance(dout, int) else (total, *dout[1:])
				layer_info.push('dout', pout, silent=True, overwrite=True)
				distrib_layer = layer_info.pull_self()
				self._param_sizes = param_sizes
		else:
			if force_distrib_layer or len(param_sizes) > 1 or total != chn:
				layers = OrderedDict()
				for key, size in zip(self._param_constraints, param_sizes):
					pout = size if isinstance(dout, int) else (size, *dout[1:])
					layer_info.push('dout', pout, silent=True, overwrite=True)
					layers[key]  = layer_info.pull_self()
				distrib_layer = nn.ModuleDict(layers)
				self._param_sizes = None
				
		return distrib_layer
	
	
	def _compute_distrib_param_sizes(self, dout):
		return [util.get_distrib_param_size(self._distrib_cls, key, dout) for key in self._param_constraints]
		
		
	def _autosample(self, dis, sample_type=None):
		'''

		:param dis: distribution
		:param sample_type: str, "best" -> select mode, else -> rsample
		:return:
		'''
		return dis.bsample() if sample_type == 'best' else dis.rsample()
	
	
	def forward(self, *args, **kwargs):
		params = super().forward(*args, **kwargs)
		dis = self._process_distrib_params(params)
		if self._use_autosample is not None:
			return self._autosample(dis, sample_type=self._use_autosample)
		return dis
	
	
	def _process_distrib_params(self, params):
		
		if self._param_sizes is not None:
			if self.distrib_layer is not None:
				params = self.distrib_layer(params)
			params = dict(zip(self._param_constraints.keys(), params.split(self._param_sizes, dim=1)))
		else:
			params = {key:self.distrib_layer[key](params) for key in self._param_constraints}
		
		params = {key: util.constrain_real(cons, params[key]) for key, cons in self._param_constraints.items()}
		return self._distrib_cls(**params)



@fig.AutoModifier('normal')
class Normal(Distribution):
	def __init__(self, A, distrib_cls=None, **kwargs):
		super().__init__(A, distrib_cls=util.get_distribution_base('normal'), **kwargs)

@fig.AutoModifier('beta')
class Beta(Distribution):
	def __init__(self, A, distrib_cls=None, **kwargs):
		super().__init__(A, distrib_cls=util.get_distribution_base('beta'), **kwargs)
#
#
# @fig.AutoModifier('kumaraswamy')
# class Kumaraswamy(Distribution):
# 	def __init__(self, A, distrib_cls=None, **kwargs):
# 		super().__init__(A, distrib_cls=util.get_distribution_base('kumaraswamy'), **kwargs)





# endregion

