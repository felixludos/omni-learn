import numpy as np
from torch import nn
import torch.optim as O
from omnibelt import unspecified_argument, agnosticmethod

from omnidata.framework import spaces
from omnidata.framework.base import Function
from omnidata.framework.hyperparameters import hparam, Parametrized
# from omnidata.framework.models import Model
from omnidata.framework.building import Builder, get_builder, register_builder


@register_builder('nonlinearity')
class BasicNonlinearlity(Builder):
	known_nonlinearities = ('relu', 'prelu', 'lrelu', 'tanh', 'softplus', 'sigmoid', 'elu', 'selu')

	ident = hparam('elu', space=known_nonlinearities)
	inplace = hparam(True, space=spaces.Binary())

	@classmethod
	def _build(cls, ident, inplace=True, **kwargs):
		if ident is None:
			return None

		if ident == 'prelu':
			return nn.PReLU(**kwargs)
		elif ident == 'elu':
			return nn.ELU(inplace=inplace, **kwargs)
		elif ident == 'lrelu':
			return nn.LeakyReLU(**kwargs)
		elif ident == 'relu':
			return nn.ReLU(inplace=inplace)
		elif ident == 'tanh':
			return nn.Tanh()
		elif ident == 'softplus':
			return nn.Softplus(**kwargs)
		elif ident == 'sigmoid':
			return nn.Sigmoid()
		elif ident == 'selu':
			return nn.SELU(inplace=inplace)
		raise cls.UnknownNonlin(ident)

	class UnknownNonlin(NotImplementedError):
		pass


@register_builder('normalization')
class BasicNormalization(Builder):
	# known_nonlinearities = ('batch', 'instance', 'layer', 'group')
	known_nonlinearities = ('batch', 'instance', 'group')

	ident = hparam('batch', space=known_nonlinearities)
	# lazy = hparam(True, space=spaces.Binary())

	width = hparam(None)
	spatial_dims = hparam(1, space=[1, 2, 3])

	num_groups = hparam(None)

	momentum = hparam(0.1, space=spaces.Bound(0.01, 0.99))
	eps = hparam(1e-5)
	affine = hparam(True, space=spaces.Binary())

	@classmethod
	def _build(cls, ident, width, spatial_dims=1, num_groups=None, lazy=True, momentum=0.1,
	           eps=1e-5, affine=True):
		if ident is None:
			return None

		norms = {
			'batch': {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d},
			'instance': {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d},
		}
		norm_fn = norms.get(ident, {}).get(spatial_dims, None)

		if ident == 'group':
			if num_groups is None:
				num_groups = 8 if width >= 16 else 4
			return nn.GroupNorm(num_groups, width, eps=eps, affine=affine)
		elif norm_fn is not None:
			return norm_fn(width, momentum=momentum, eps=eps, affine=affine)
		else:
			raise cls.UnknownNorm(ident)

	class UnknownNorm(NotImplementedError):
		pass


@register_builder('loss')
class BasicLoss(Builder):
	target_space = hparam(required=True)

	@classmethod
	def _build(cls, target_space):
		if isinstance(target_space, spaces.Categorical):
			return nn.CrossEntropyLoss()
		elif isinstance(target_space, spaces.Continuous):
			return nn.MSELoss()
		else:
			raise cls.UnknownTarget(target_space)

	class UnknownTarget(NotImplementedError):
		pass


class Reshaper(nn.Module):  # by default flattens
	def __init__(self, dout=(-1,)):
		super().__init__()
		self.dout = dout

	def extra_repr(self):
		return f'out={self.dout}'

	def forward(self, x):
		B = x.size(0)
		return x.view(B, *self.dout)


@register_builder('mlp')
class MLP(Builder, Function, nn.Sequential):
	def __init__(self, layers=None, **kwargs):
		# self.get_hparam('nonlin').space = get_builder('nonlinearity').get_hparam('ident').space
		# self.get_hparam('norm').space = get_builder('normalization').get_hparam('ident').space
		if layers is None:
			kwargs = self._extract_hparams(kwargs)
			layers = self._build_layers()
		super().__init__(layers, **kwargs)
		self._nonlin_builder = get_builder('nonlinearity')()
		self._norm_builder = get_builder('normalization')()

	din = hparam(required=True)
	dout = hparam(required=True)

	hidden = hparam(())

	nonlin = hparam('elu', ref=get_builder('nonlinearity').get_hparam('ident'))
	norm = hparam(None, ref=get_builder('normalization').get_hparam('ident'))
	dropout = hparam(0., space=spaces.Bound(0., 0.5))

	bias = hparam(True)

	out_nonlin = hparam(None)
	out_norm = hparam(None)
	out_bias = hparam(True)

	@agnosticmethod
	def _expand_dim(self, dim):
		if isinstance(dim, int):
			dim = [dim]
		if isinstance(dim, (list, tuple)):
			return dim
		if isinstance(dim, spaces.Dim):
			return dim.expanded_shape
		raise NotImplementedError(dim)

	def _create_nonlin(self, ident=None, inplace=unspecified_argument, **kwargs):
		return self._nonlin_builder.build(ident, inplace=inplace, **kwargs)

	def _create_norm(self, norm, width, **kwargs):
		return self._norm_builder.build(norm, width, spatial_dim=1, **kwargs)

	def _create_linear_layer(self, din, dout, bias=True):
		return nn.Linear(din, dout, bias=bias)

	@agnosticmethod
	def _build_layer(self, din, dout, nonlin=unspecified_argument, norm=unspecified_argument,
	                 dropout=None, bias=unspecified_argument):

		if nonlin is unspecified_argument:
			nonlin = self.nonlin

		if norm is unspecified_argument:
			norm = self.norm

		if bias is unspecified_argument:
			bias = self.bias

		in_shape = self._expand_dim(din)
		in_width = int(np.product(in_shape))

		layers = []

		if len(in_shape) > 1:
			layers.append(nn.Flatten())

		out_shape = self._expand_dim(dout)
		out_width = int(np.product(in_shape))

		layers.append(self._create_linear_layer(in_width, out_width, bias=bias))

		norm = self._create_norm(norm, width=out_width)
		if norm is not None:
			layers.append(norm)

		nonlin = self._create_nonlin(nonlin)
		if nonlin is not None:
			layers.append(nonlin)

		if dropout is not None and dropout > 0:
			layers.append(nn.Dropout(dropout))

		if len(out_shape) > 1:
			layers.append(Reshaper(out_shape))
		return dout, layers

	@agnosticmethod
	def _build_outlayer(self, din, dout, nonlin=unspecified_argument, norm=unspecified_argument,
	                    dropout=None, bias=unspecified_argument, **kwargs):
		if nonlin is unspecified_argument:
			nonlin = self.out_nonlin
		if norm is unspecified_argument:
			norm = self.out_norm
		if bias is unspecified_argument:
			bias = self.out_bias
		return self._build_layer(din, dout, nonlin=nonlin, norm=norm, dropout=dropout, bias=bias, **kwargs)[1]

	@agnosticmethod
	def _build_layers(self, din, dout, hidden=()):
		layers = []
		start_dim, end_dim = din, din
		for end_dim in hidden:
			start_dim, new_layers = self._build_layer(start_dim, end_dim)
			layers.extend(new_layers)
		layers.extend(self._build_outlayer(end_dim, dout))
		return layers

	@classmethod
	def _build(cls, din, dout, hidden, nonlin, norm, bias=True, **kwargs):
		return cls(din=din, dout=dout, hidden=hidden, nonlin=nonlin, norm=norm, bias=bias, **kwargs)




# from omnibelt import agnosticmethod, unspecified_argument
# import numpy as np
# from torch import nn
#
# from omnidata.framework import hparam, Function, Model, spaces
#
#
#
# class Reshaper(nn.Module): # by default flattens
#
# 	def __init__(self, dout=(-1,)):
# 		super().__init__()
#
# 		self.dout = dout
#
#
# 	def extra_repr(self):
# 		return f'out={self.dout}'
#
#
# 	def forward(self, x):
# 		B = x.size(0)
# 		return x.view(B, *self.dout)
#
#
#
# class MLP(Model, Function, nn.Sequential):
# 	# def __init__(self, layers, din=None, dout=None, **kwargs):
# 	# 	super().__init__(layers, din=din, dout=dout, **kwargs)
#
# 	nonlin = hparam('elu', space=['prelu', 'lrelu', 'relu', 'tanh',
# 	                              'softplus', 'sigmoid', 'elu', 'selu'])
# 	nonlin_inplace = hparam(True)
# 	nonlin_kwargs = hparam({})
#
# 	norm = hparam(None, space=['batch', 'instance'])
# 	norm_kwargs = hparam({})
#
# 	bias = hparam(True)
#
# 	out_nonlin = hparam(None)
# 	out_norm = hparam(None)
# 	out_bias = hparam(True)
#
# 	hidden = hparam(())
#
#
# 	class Builder(Model.Builder):
# 		@agnosticmethod
# 		def _expand_dim(self, dim):
# 			if isinstance(dim, int):
# 				dim = [dim]
# 			if isinstance(dim, (list, tuple)):
# 				return dim
# 			if isinstance(dim, spaces.Dim):
# 				return dim.expanded_shape
# 			raise NotImplementedError(dim)
#
#
# 		def _create_nonlin(self, ident=None, inplace=unspecified_argument, **_kwargs):
# 			# if ident is unspecified_argument:
# 			# 	ident = self.nonlin
# 			if inplace is unspecified_argument:
# 				inplace = self.nonlin_inplace
# 			kwargs = self.nonlin_kwargs.copy()
# 			kwargs.update(_kwargs)
#
# 			if ident is None:
# 				return None
#
# 			if ident == 'prelu':
# 				return nn.PReLU(**kwargs)
# 			elif ident == 'lrelu':
# 				return nn.LeakyReLU(**kwargs)
# 			elif ident == 'relu':
# 				return nn.ReLU(inplace=inplace)
# 			elif ident == 'tanh':
# 				return nn.Tanh()
# 			elif ident == 'softplus':
# 				return nn.Softplus(**kwargs)
# 			elif ident == 'sigmoid':
# 				return nn.Sigmoid()
# 			elif ident == 'elu':
# 				return nn.ELU(inplace=inplace, **kwargs)
# 			elif ident == 'selu':
# 				return nn.SELU(inplace=inplace)
# 			raise self.UnknownNonlin(ident)
#
#
# 		class UnknownNonlin(NotImplementedError):
# 			pass
#
#
# 		def _create_norm(self, norm, width, **_kwargs):
# 			# if norm is unspecified_argument:
# 			# 	norm = self.norm
# 			# if width is unspecified_argument:
# 			# 	width = self.dout.expanded_len
# 			kwargs = self.norm_kwargs.copy()
# 			kwargs.update(_kwargs)
#
# 			if norm is None:
# 				return None
#
# 			if norm == 'batch':
# 				return nn.BatchNorm1d(width, **kwargs)
# 			if norm == 'instance':
# 				return nn.InstanceNorm1d(width, **kwargs)
# 			raise self.UnknownNorm
#
#
# 		class UnknownNorm(NotImplementedError):
# 			pass
#
#
# 		@agnosticmethod
# 		def _build_layer(self, din, dout, nonlin=unspecified_argument, norm=unspecified_argument,
# 		                 bias=unspecified_argument):
#
# 			if nonlin is unspecified_argument:
# 				nonlin = self.nonlin
#
# 			if norm is unspecified_argument:
# 				norm = self.norm
#
# 			if bias is unspecified_argument:
# 				bias = self.bias
#
# 			in_shape = self._expand_dim(din)
# 			in_width = int(np.product(in_shape))
#
# 			layers = []
#
# 			if len(in_shape) > 1:
# 				layers.append(nn.Flatten())
#
# 			out_shape = self._expand_dim(dout)
# 			out_width = int(np.product(in_shape))
#
# 			layers.append(nn.Linear(in_width, out_width, bias=bias))
#
# 			nonlin = self._create_nonlin(nonlin)
# 			if nonlin is not None:
# 				layers.append(nonlin)
#
# 			norm = self._create_norm(norm, width=out_width)
# 			if norm is not None:
# 				layers.append(norm)
#
# 			if len(out_shape) > 1:
# 				layers.append(Reshaper(out_shape))
# 			return dout, layers
#
#
# 		@agnosticmethod
# 		def _build_outlayer(self, din, dout, nonlin=unspecified_argument, norm=unspecified_argument,
# 		                    bias=unspecified_argument):
# 			if nonlin is unspecified_argument:
# 				nonlin = self.out_nonlin
# 			if norm is unspecified_argument:
# 				norm = self.out_norm
# 			if bias is unspecified_argument:
# 				bias = self.out_bias
# 			return self._build_layer(din, dout, nonlin=nonlin, norm=norm, bias=bias)[1]
#
#
# 		@agnosticmethod
# 		def _build_layers(self, din, dout, hidden=()):
# 			layers = []
# 			start_dim, end_dim = din, din
# 			for end_dim in hidden:
# 				start_dim, new_layers = self._build_layer(start_dim, end_dim)
# 				layers.extend(new_layers)
# 			layers.extend(self._build_outlayer(end_dim, dout))
# 			return layers
#
#
# 		def build(self, kwargs=None):
# 			if kwargs is None:
# 				kwargs = dict(layers=self._build_layers(din=self.din, dout=self.dout, hidden=self.hidden),
# 			                     din=self.din, dout=self.dout)
# 			return super().build(kwargs)














