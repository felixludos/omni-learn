import numpy as np
from torch import nn
import torch.optim as O
from omnibelt import unspecified_argument, agnostic

import omnifig as fig

from omnidata import Function, spaces
from omnidata import hparam, Parameterized, \
	get_builder, machine, Machine, RegistryBuilder, BasicBuilder, Builder, Buildable, with_hparams
from . import base as reg


@reg.builder('nonlin')
@reg.builder('nonlinearity')
class BasicNonlinearlity(RegistryBuilder, default_ident='elu', products={
							'relu': nn.ReLU,
							'prelu': nn.PReLU,
							'lrelu': nn.LeakyReLU,
							'tanh': nn.Tanh,
							'softplus': nn.Softplus,
							'sigmoid': nn.Sigmoid,
							'elu': nn.ELU,
							'selu': nn.SELU,
                         }):
	inplace = hparam(True, space=spaces.Binary())

	@agnostic
	def build(self, ident, inplace, **kwargs):
		product = self.product(ident=ident, inplace=inplace, **kwargs)
		if product in {nn.ELU, nn.ReLU, nn.SELU}:
			return product(inplace=inplace, **kwargs)
		return product(**kwargs)



@reg.builder('norm')
@reg.builder('normalization')
class BasicNormalization(RegistryBuilder, default_ident='batch', products={
							'batch1d': nn.BatchNorm1d,
							'batch2d': nn.BatchNorm2d,
							'batch3d': nn.BatchNorm3d,
							'instance1d': nn.InstanceNorm1d,
							'instance2d': nn.InstanceNorm2d,
							'instance3d': nn.InstanceNorm3d,
							'group': nn.GroupNorm,
                         }):

	width = hparam(None)
	spatial_dims = hparam(1, space=[1, 2, 3])

	num_groups = hparam(None)

	momentum = hparam(0.1, space=spaces.Bound(0.01, 0.99))
	eps = hparam(1e-5)
	affine = hparam(True, space=spaces.Binary())

	@agnostic
	def product(self, ident, spatial_dims=1, **kwargs):
		if ident in {'batch', 'instance'}:
			ident = f'{ident}{spatial_dims}d'
		return super().product(ident, **kwargs)

	@agnostic
	def build(self, ident, width, spatial_dims=1, num_groups=None, momentum=0.1,
	           eps=1e-5, affine=True, **kwargs):
		
		product = self.product(ident=ident, spatial_dims=spatial_dims, **kwargs)

		if issubclass(product, nn.GroupNorm):
			if num_groups is None:
				num_groups = 8 if width >= 16 else 4
			return product(num_groups, width, eps=eps, affine=affine)
		return product(width, momentum=momentum, eps=eps, affine=affine, **kwargs)



@reg.builder('loss')
class BasicLoss(Builder):
	target_space = hparam(required=True)
	
	@agnostic
	def product_base(self, target_space):
		if isinstance(target_space, spaces.Categorical):
			return nn.CrossEntropyLoss
		elif isinstance(target_space, spaces.Continuous):
			return nn.MSELoss
		else:
			raise self.NoProductFound(target_space)
		


@reg.component('reshaper')
class Reshaper(Buildable, nn.Module):  # by default flattens
	def __init__(self, dout=(-1,)):
		super().__init__()
		self.dout = dout

	def extra_repr(self):
		return f'out={self.dout}'

	def forward(self, x):
		B = x.size(0)
		return x.view(B, *self.dout)



@reg.builder('linear')
class BasicLinear(Builder):
	width = hparam(None)
	
	din = hparam(None)
	dout = hparam(None)

	bias = hparam(True, space=spaces.Binary())

	@agnostic
	def product_base(self, *args, **kwargs):
		return nn.Linear

	@agnostic
	def build(self, width: int = None, *, din=None, dout=None, bias=True, **kwargs):
		if din is None and dout is None:
			raise ValueError('Must specify either din or dout')
		if width is not None and din is not None and dout is not None:
			raise ValueError('Cannot specify width and both din and dout')

		if isinstance(din, spaces.Dim):
			din = din.width
		if isinstance(dout, spaces.Dim):
			dout = dout.width

		if width is not None:
			if din is None:
				din = width
			elif dout is None:
				dout = width

		return self.product()(din, dout, bias=bias, **kwargs)



@reg.builder('dropout')
class BasicDropout(Builder): # TODO: enable multiple dims
	p = hparam(0.1, space=spaces.Bound(0, 1))

	@agnostic
	def product_base(self, *args, **kwargs):
		return nn.Dropout

	@agnostic
	@fig.config_aliases(p='prob')
	def build(self, p=0.1, **kwargs):
		return self.product()(p=p, **kwargs)



@reg.builder('dense-layer')
class DenseLayer(Buildable, nn.Module):
	
	linear = machine(builder='linear')

	norm = machine(None, builder='normalization')
	nonlin = machine('elu', builder='nonlinearity')
	dropout = machine(None, builder='dropout')

	@hparam
	def din(self):
		return self.linear.din
	@hparam
	def dout(self):
		return self.linear.dout

	def forward(self, x):
		x = self.linear(x)
		if self.norm is not None:
			x = self.norm(x)
		if self.nonlin is not None:
			x = self.nonlin(x)
		if self.dropout is not None:
			x = self.dropout(x)
		return x



@reg.builder('mlp')
class MLP(Buildable, Function, nn.Sequential):
	def __init__(self, layers=None, **kwargs):
		if layers is None:
			kwargs = self._extract_hparams(kwargs)
			layers = self._default_build_layers()
		super().__init__(*layers, **kwargs)

	din = hparam(required=True)
	dout = hparam(required=True)

	hidden = hparam(())

	dropout = hparam(0., space=spaces.Bound(0., 1.))

	nonlin = hparam('elu', space=get_builder('nonlinearity').get_hparam('ident').space)
	out_nonlin = hparam(None, space=get_builder('nonlinearity').get_hparam('ident').space)

	norm = hparam(None, space=get_builder('normalization').get_hparam('ident').space)
	out_norm = hparam(None, space=get_builder('normalization').get_hparam('ident').space)

	bias = hparam(True, space=spaces.Binary())
	out_bias = hparam(True, space=spaces.Binary())

	@agnostic
	def _expand_dim(self, dim):
		if isinstance(dim, int):
			dim = [dim]
		if isinstance(dim, (list, tuple)):
			return dim
		if isinstance(dim, spaces.Dim):
			return dim.expanded_shape
		raise NotImplementedError(dim)

	@agnostic
	def _create_nonlin(self, ident=None, **kwargs):
		if ident is None:
			return None
		return get_builder('nonlinearity').build(ident, **kwargs)

	@agnostic
	def _create_norm(self, norm, width, spatial_dim=1, **kwargs):
		if norm is None:
			return None
		return get_builder('normalization').build(norm, width, spatial_dim=spatial_dim, **kwargs)

	@agnostic
	def _create_linear_layer(self, din, dout, bias=True):
		return get_builder('linear').build(din=din, dout=dout, bias=bias)

	@agnostic
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
		out_width = int(np.product(out_shape))

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

	@agnostic
	def _build_outlayer(self, din, dout, nonlin=unspecified_argument, norm=unspecified_argument,
	                    dropout=None, bias=unspecified_argument, **kwargs):
		if nonlin is unspecified_argument:
			nonlin = self.out_nonlin
		if norm is unspecified_argument:
			norm = self.out_norm
		if bias is unspecified_argument:
			bias = self.out_bias
		return self._build_layer(din, dout, nonlin=nonlin, norm=norm, dropout=dropout, bias=bias, **kwargs)[1]

	@agnostic
	def _default_build_layers(self):
		layers = []
		start_dim, end_dim = self.din, self.din
		for end_dim in self.hidden:
			start_dim, new_layers = self._build_layer(start_dim, end_dim)
			layers.extend(new_layers)
		layers.extend(self._build_outlayer(end_dim, self.dout))
		return layers



# # @reg.builder('qik-mlp')
# class QikMLP(Buildable, nn.Module):
# 	din = hparam(required=True)
# 	dout = hparam(required=True)
#
# 	hidden = hparam(())
#
# 	layer_builder = machine(builder='dense-layer', cache=False)
#
# 	nonlin = machine('elu', builder='nonlinearity', cache=False)
# 	out_nonlin = machine(None, builder='nonlinearity', cache=False)
#
# 	norm = machine(None, builder='normalization', cache=False)
# 	out_norm = machine(None, builder='normalization', cache=False)
#
# 	bias = hparam(True, space=spaces.Binary())
# 	out_bias = hparam(True, space=spaces.Binary())
#
# 	# TODO: finish this
#
# 	@agnostic
# 	def _expand_dim(self, dim):
# 		if isinstance(dim, int):
# 			dim = [dim]
# 		if isinstance(dim, (list, tuple)):
# 			return dim
# 		if isinstance(dim, spaces.Dim):
# 			return dim.expanded_shape
# 		raise NotImplementedError(dim)
#
# 	@agnostic
# 	def _build_layer(self, din, dout):
# 		in_shape = self._expand_dim(din)
# 		in_width = int(np.product(in_shape))
#
# 		layers = []
#
# 		if len(in_shape) > 1:
# 			layers.append(nn.Flatten())
#
# 		out_shape = self._expand_dim(dout)
# 		out_width = int(np.product(in_shape))
#
# 		layers.append(self._create_linear_layer(in_width, out_width, bias=bias))
#
# 		norm = self._create_norm(norm, width=out_width)
# 		if norm is not None:
# 			layers.append(norm)
#
# 		nonlin = self._create_nonlin(nonlin)
# 		if nonlin is not None:
# 			layers.append(nonlin)
#
# 		if dropout is not None and dropout > 0:
# 			layers.append(nn.Dropout(dropout))
#
# 		if len(out_shape) > 1:
# 			layers.append(Reshaper(out_shape))
# 		return dout, layers
#
# 	@agnostic
# 	def _build_outlayer(self, din, dout, nonlin=unspecified_argument, norm=unspecified_argument,
# 	                    dropout=None, bias=unspecified_argument, **kwargs):
# 		if nonlin is unspecified_argument:
# 			nonlin = self.out_nonlin
# 		if norm is unspecified_argument:
# 			norm = self.out_norm
# 		if bias is unspecified_argument:
# 			bias = self.out_bias
# 		return self._build_layer(din, dout, nonlin=nonlin, norm=norm, dropout=dropout, bias=bias, **kwargs)[1]
#
# 	@agnostic
# 	def _build_layers(self, din, dout, hidden=()):
# 		layers = []
# 		start_dim, end_dim = din, din
# 		for end_dim in hidden:
# 			start_dim, new_layers = self._build_layer(start_dim, end_dim)
# 			layers.extend(new_layers)
# 		layers.extend(self._build_outlayer(end_dim, dout))
# 		return layers
#
# 	@agnostic
# 	def build(self, din, dout, hidden, nonlin, norm, bias=True, **kwargs):
# 		return super().build(din=din, dout=dout, hidden=hidden, nonlin=nonlin, norm=norm, bias=bias, **kwargs)




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














