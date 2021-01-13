
import torch
from torch import nn

import omnifig as fig


# from .. import framework as fm
from ..op import framework as fm
from .nets import MultiLayer

#
#
# class StorablePrior(fm.FunctionBase):
# 	def __init__(self, din=None, dout=None, prior_dim=None):
# 		super().__init__(din, dout)
# 		self.prior_dim = prior_dim
#
# 		self.volatile.prior = None
#
# 	def get_style_dim(self):
# 		return self.prior_dim
#
# 	def set_prior(self, p):
# 		self.volatile.prior = p
#
# 	def check_stored(self, N=1):
# 		if self.volatile.prior is not None:
# 			p = self.volatile.prior
# 			self.volatile.prior = None
# 			assert p.size(0) == N
# 			return p
# 		return self.sample_prior(N)
#
# 	def sample_prior(self, N=1):
# 		return torch.randn(N, self.prior_dim, device=self.device)
#
# @fig.Component('adain')
# class AdaIN(StorablePrior, fm.FunctionBase):
#
# 	def __init__(self, A):
# 		channels = None
# 		din = A.pull('din', None)
# 		dout = None
# 		if din is None:
# 			dout = A.pull('dout', None)
#
# 			if dout is None:
# 				channels = A.pull('channels')
# 			else:
# 				din = dout
# 				channels = dout[0]
# 		else:
# 			dout = din
#
# 			channels = dout[0]
#
# 		assert channels is not None, 'no info'
#
# 		norm = A.pull('normalize', False)
#
# 		style_dim = A.pull('style_dim', '<>latent_dim')
#
# 		A.push('net._type', 'mlp', overwrite=False, silent=True)
# 		A.push('net.din', style_dim, silent=True)
# 		A.push('net.dout', channels*2, silent=True)
# 		net = A.pull('net')
#
# 		super().__init__(din, dout, style_dim)
#
# 		self.feature_dim = channels
#
# 		self.net = net
#
# 		self.normalize = nn.InstanceNorm2d(channels, affine=False) if norm else None
#
# 	# def extra_repr(self):
# 	# 	return f'style={self.get_style_dim()}, normalize={self.normalize is not None}'
#
# 	def process_prior(self, p):
# 		return self.net(p)
#
# 	def process_features(self, x):
# 		return x if self.normalize is None else self.normalize(x)
#
# 	def forward(self, x, p=None):
#
# 		c = self.process_features(x)
#
# 		if p is None:
# 			p = self.check_stored(x.size(0))
# 		q = self.process_prior(p)
# 		m = q.narrow(1, 0, self.feature_dim).unsqueeze(-1).unsqueeze(-1)
# 		s = q.narrow(1,self.feature_dim, self.feature_dim).unsqueeze(-1).unsqueeze(-1).exp()
#
# 		return s*c + m
#
#
# @fig.Component('style')
# class StyleModel(MultiLayer, StorablePrior):
#
# 	def _create_layers(self, A):
#
# 		split_style = A.pull('split-style', False)
#
# 		layers = super()._create_layers(A)
#
# 		branches = [layer for layer in layers if isinstance(layer, StorablePrior)]
# 		bdims = [branch.get_style_dim() for branch in branches]
#
# 		if split_style:
# 			style_dim = sum(bdims)
# 		else:
# 			style_dim = bdims[0]
# 			assert all(b == style_dim for b in bdims)
#
# 		self.split_style = split_style
# 		self.branches = branches
# 		self.branch_dims = bdims
# 		self.style_dim = style_dim
# 		self.prior_dim = style_dim
#
# 		return layers
#
# 	def forward(self, x, z=None):
#
# 		if z is None:
# 			z = self.check_stored(x.size(0))
#
# 		if self.split_style:
# 			priors = z.split(self.branch_dims, dim=1)
# 			for branch, prior in zip(self.branches, priors):
# 				branch.set_prior(prior)
# 		else:
# 			for branch in self.branches:
# 				branch.set_prior(z)
#
# 		return super().forward(x)
#




