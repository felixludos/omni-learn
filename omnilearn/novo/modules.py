import numpy as np
from torch import nn
import torch.optim as O
from omnibelt import unspecified_argument, agnosticmethod

from omnidata.framework import spaces
from omnidata.framework.base import Function
# from omnidata.framework.models import Model
from omnidata.framework.hyperparameters import hparam, Parameterized
from omnidata.framework.building import Builder, get_builder, register_builder, ClassBuilder
from omnidata.framework.machines import machine, MachineParametrized


@register_builder('nonlin')
@register_builder('nonlinearity')
class BasicNonlinearlity(ClassBuilder, default_ident='elu'):
	inplace = hparam(True, space=spaces.Binary())
	

	@agnosticmethod
	def product_registry(self):
		return {
			'relu': nn.ReLU,
			'prelu': nn.PReLU,
			'lrelu': nn.LeakyReLU,
			'tanh': nn.Tanh,
			'softplus': nn.Softplus,
			'sigmoid': nn.Sigmoid,
			'elu': nn.ELU,
			'selu': nn.SELU,
			**super().product_registry()
		}
	
	
	@agnosticmethod
	def _build(self, ident, inplace=True, **kwargs):
		product = self.product(ident=ident, inplace=inplace, **kwargs)
		if product in {nn.ELU, nn.ReLU, nn.SELU}:
			return product(inplace=inplace, **kwargs)
		return product(**kwargs)



@register_builder('norm')
@register_builder('normalization')
class BasicNormalization(ClassBuilder, default_ident='batch'):

	@agnosticmethod
	def product_registry(self):
		return {
			'batch1d': nn.BatchNorm1d,
			'batch2d': nn.BatchNorm2d,
			'batch3d': nn.BatchNorm3d,
			'instance1d': nn.InstanceNorm1d,
			'instance2d': nn.InstanceNorm2d,
			'instance3d': nn.InstanceNorm3d,
			'group': nn.GroupNorm,
			**super().product_registry()
		}

	# lazy = hparam(True, space=spaces.Binary())

	width = hparam(None)
	spatial_dims = hparam(1, space=[1, 2, 3])

	num_groups = hparam(None)

	momentum = hparam(0.1, space=spaces.Bound(0.01, 0.99))
	eps = hparam(1e-5)
	affine = hparam(True, space=spaces.Binary())


	@agnosticmethod
	def _product(self, ident, spatial_dims=1, **kwargs):
		if ident in {'batch', 'instance'}:
			ident = f'{ident}{spatial_dims}d'
		return super()._product(ident, **kwargs)


	@agnosticmethod
	def _build(self, ident, width, spatial_dims=1, num_groups=None, momentum=0.1,
	           eps=1e-5, affine=True, **kwargs):
		
		product = self.product(ident=ident, spatial_dims=spatial_dims, **kwargs)

		if product is nn.GroupNorm:
			if num_groups is None:
				num_groups = 8 if width >= 16 else 4
			return product(num_groups, width, eps=eps, affine=affine)
		return product(width, momentum=momentum, eps=eps, affine=affine, **kwargs)



@register_builder('loss')
class BasicLoss(Builder):
	target_space = hparam(required=True)
	
	@agnosticmethod
	def _product(self, target_space):
		if isinstance(target_space, spaces.Categorical):
			return nn.CrossEntropyLoss
		elif isinstance(target_space, spaces.Continuous):
			return nn.MSELoss
		else:
			raise self.NoProductFound(target_space)
		


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
		if layers is None:
			kwargs = self._extract_hparams(kwargs)
			layers = self._build_layers()
		super().__init__(layers, **kwargs)

	din = hparam(required=True)
	dout = hparam(required=True)

	hidden = hparam(())

	dropout = hparam(0., space=spaces.Bound(0., 0.5))

	nonlin = machine('elu', builder='nonlinearity')
	out_nonlin = machine(None, builder='nonlinearity')

	norm = machine(None, builder='normalization')
	out_norm = machine(None, builder='normalization')

	bias = hparam(True)
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

	@agnosticmethod
	def _create_nonlin(self, ident=None, **kwargs):
		if ident is None:
			return None
		return self.get_hparam('nonlin').get_builder().build(ident=ident, **kwargs)

	@agnosticmethod
	def _create_norm(self, norm, width, spatial_dim=1, **kwargs):
		if norm is None:
			return None
		return self.get_hparam('norm').get_builder().build(norm, width=width, spatial_dim=spatial_dim, **kwargs)

	@agnosticmethod
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

	@agnosticmethod
	def _build(self, din, dout, hidden, nonlin, norm, bias=True, **kwargs):
		return super()._build(din=din, dout=dout, hidden=hidden, nonlin=nonlin, norm=norm, bias=bias, **kwargs)
		



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













