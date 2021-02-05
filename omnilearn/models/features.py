
import torch
from torch import nn
from torch.distributions import Normal as NormalDistribution

import omnifig as fig

# import torch
# import torch.nn as nn
# from .. import framework as fm
# from ..op import framework as fm

# from omnibelt import InitWall, unspecified_argument
# import omnifig as fig

# from .layers import make_MLP

from ..op.framework import Function
from .nets import MultiLayer
from .. import util

###############
# region Prior
###############

class Prior(Function):

	def __init__(self, A, prior_dim=None, **kwargs):
		super().__init__(A, **kwargs)
		# if prior_dim is None:
		# 	prior_dim = A.pull('latent-dim', '<>din', None, silent=True)
		self.prior_dim = prior_dim

	def sample_prior(self, *shape):
		raise NotImplementedError

@fig.AutoModifier('gaussian-prior')
class Gaussian(Prior):
	def sample_prior(self, N=1, *D):
		if not len(D):
			D = self.prior_dim
			if isinstance(D, int):
				D = (D,)
		return torch.randn(N, *D, device=self.device)


@fig.AutoModifier('uniform-prior')
class Uniform(Prior):
	def sample_prior(self, N=1, *D):
		if not len(D):
			D = self._prior_dim
			if isinstance(D, int):
				D = (D,)
		return torch.rand(N, *D, device=self.device)


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

	def sample_prior(self, N=1):
		q = super().sample_prior(N)
		return q if self.prior_tfm is None else self.prior_tfm(q)


# endregion


###############
# region Normal
###############

@fig.AutoModifier('normal')
class Normal(Function):
	def __init__(self, A, **kwargs):
		
		dout = None

		change_dout = A.pull('change-dout', True)
		if change_dout:
			dout_key = A.pull('_dout_key', None)
			if dout_key is None:
				dout = A.pull('_dout', None)
				if dout is not None:
					dout_key = '_dout'
				else:
					dout = A.pull('dout')
			else:
				dout = A.pull(dout_key)
			
			if dout is not None:

				if isinstance(dout, int):
					chn = dout * 2
					dout = chn

				else:
					chn = dout[0]
					chn = chn * 2
					dout = (chn, *dout[1:])
				
				if dout_key is None:
					A.push('dout', dout)
					# A.push('dout', dout, silent=True)
				else:
					A.push(dout_key, dout, silent=True)

		super().__init__(A, **kwargs)
		
		if dout is None:
			dout = self.dout
		self.full_dout = dout

		split = change_dout
		if not change_dout:
			split = A.pull('split-out', True)

		out = None
		if split:

			if isinstance(dout, int):
				assert dout % 2 == 0
				chn = dout // 2
				dout = chn

			else:
				chn = dout[0]
				assert chn % 2 == 0
				chn = chn // 2
				dout = (chn, *dout[1:])

		else:

			if isinstance(dout, int):
				chn = dout

				out = nn.Linear(chn, chn * 2)

			# A.push('out-layer._type', 'dense-layer', silent=True)
			# A.push('')

			else:
				chn = dout[0]

				out = nn.Conv2d(chn, chn * 2, kernel_size=1)

		# A.push('out-layer._type', 'conv-layer', silent=True)

		self.normal_layer = out
		self.width = chn
		self.dout = dout

	def forward(self, *args, **kwargs):

		x = super().forward(*args, **kwargs)

		if self.normal_layer is not None:
			x = self.normal_layer(x)

		mu, sigma = x.narrow(1, 0, self.width), x.narrow(1, self.width, self.width).exp()

		return NormalDistribution(mu, sigma)

# endregion

