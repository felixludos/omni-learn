
import torch

import omnifig as fig

from ..op.framework import Function
from .. import util

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