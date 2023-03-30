from typing import Iterable, Any, Optional, Union, Callable, Tuple, Dict, List
import numpy as np
from torch import nn
import torch.optim as O
from omnibelt import unspecified_argument, agnostic

import omnifig as fig

from omnidata import Function, spaces

from .base import hparam, inherit_hparams, submodule, submachine, material, space, indicator, machine, \
	Structured, Builder
from . import base as reg



@reg.builder('nonlin')
class Activation(reg.BranchBuilder, branch='nonlin', default_ident='relu', products={
							'relu': nn.ReLU,
							'prelu': nn.PReLU,
							'lrelu': nn.LeakyReLU,
							'tanh': nn.Tanh,
							'softplus': nn.Softplus,
							'sigmoid': nn.Sigmoid,
							'elu': nn.ELU,
							'selu': nn.SELU,
                         }):
	inplace = hparam(True, space=spaces.Binary(), hidden=True)


	def _build_kwargs(self, product, ident, **kwargs):
		kwargs = super()._build_kwargs(product, ident, **kwargs)
		if issubclass(product, (nn.ELU, nn.ReLU, nn.SELU)) and 'inplace' not in kwargs:
			kwargs['inplace'] = self.inplace
		return kwargs



class CrossEntropyLoss(nn.CrossEntropyLoss):
	def forward(self, input, target):
		if target.ndim > 1:
			target = target.view(-1)
		return super().forward(input, target)



@reg.builder('criterion')
class CriterionBuilder(Builder):
	target_space = space('target')
	@space('input')
	def input_space(self):
		return self.target_space


	def product_signatures(self, *args, **kwargs):
		yield self._Signature('output', inputs=('input', 'target'))


	def product_base(self, target_space=None):
		if target_space is None:
			target_space = self.target_space
		if isinstance(target_space, spaces.Categorical):
			return nn.CrossEntropyLoss
		elif isinstance(target_space, spaces.Continuous):
			return nn.MSELoss
		else:
			raise self.NoProductFound(target_space)



@reg.component('reshaper')
class Reshaper(Structured, nn.Module):  # by default flattens
	dout_shape = hparam((-1,))


	def extra_repr(self):
		return f'out={self.dout_shape}'


	def forward(self, x):
		B = x.size(0)
		return x.view(B, *self.dout_shape)



@reg.builder('linear')
class LinearBuilder(Builder):
	din = space('input')
	dout = space('output')


	def product_base(self, *args, **kwargs):
		return nn.Linear


	def _build_kwargs(self, product, *, in_features=None, out_features=None, bias=None, **kwargs):
		kwargs = super()._build_kwargs(product, **kwargs)

		if in_features is None:
			in_features = self.din.width
		kwargs['in_features'] = in_features

		if out_features is None:
			out_features = self.dout.width
		kwargs['out_features'] = out_features

		return kwargs



@reg.builder('linear')
class BasicLinear(Builder):
	width = hparam(None)
	
	din = hparam(None)
	dout = hparam(None)

	bias = hparam(True, space=spaces.Binary())


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
	
	linear = submodule(builder='linear')

	norm = submodule(None, builder='norm')
	nonlin = submodule('elu', builder='nonlin')
	dropout = submodule(None, builder='dropout')

	@hparam(hidden=True)
	def din(self):
		return self.linear.din
	@hparam(hidden=True)
	def dout(self):
		return self.linear.dout

	def build(self, din, dout, **kwargs):
		self.linear = self.linear(din=din, dout=dout, **kwargs)
		self.norm = self.norm(width=dout, **kwargs)
		self.nonlin = self.nonlin(**kwargs)
		self.dropout = self.dropout(**kwargs)

	def forward(self, x):
		x = self.linear(x)
		if self.norm is not None:
			x = self.norm(x)
		if self.nonlin is not None:
			x = self.nonlin(x)
		if self.dropout is not None:
			x = self.dropout(x)
		return x


class Sequential(Structured, nn.Sequential):

	din = hparam(None)
	dout = hparam(None)

	def __init__(self, layers=(), din=None, dout=None, **kwargs):
		if len(layers):
			if din is None:
				din = getattr(layers[0], 'din', None)
			if dout is None:
				dout = getattr(layers[-1], 'dout', None)
		super().__init__(*layers, din=din, dout=dout, **kwargs)



@reg.builder('feedforward')
class Feedforward(Builder):
	din = hparam(None)
	dout = hparam(None)

	_product_base = Sequential

	@agnostic
	def product_base(self, *args, **kwargs):
		return self._product_base

	@agnostic
	def _build_block(self, builder, *args, **kwargs):
		builder = get_builder(builder)
		layer = builder.build(*args, **kwargs)
		return [layer], layer.din, layer.dout

	@agnostic
	def build(self, layer_builders=None, din=None, dout=None, **kwargs):
		if layer_builders is None:
			layer_builders = []
		assert din is not None or dout is not None, 'Must specify either din or dout'

		reverse = din is None
		start, end = (None, dout) if reverse else (din, None)

		layers = []
		for builder in reversed(layer_builders) if reverse else layer_builders:
			block, block_din, block_dout = self._build_block(builder, din=start, dout=end)
			layers.extend(reversed(block) if reverse else block)
			if reverse:
				end = block_din
			else:
				start = block_dout

		if reverse:
			layers = list(reversed(layers))

		ff = super().build(layers, **kwargs)
		if din is not None and dout is not None:
			assert (ff.din, ff.dout) == (din, dout), f'Feedforward has wrong shape: ' \
			                                         f'{ff.din} -> {ff.dout} != {din} -> {dout}'
		return ff



class MLP(Sequential): # just for the name
	pass



@reg.builder('mlp')
@inherit_hparams('din', 'dout')
class MLPBuilder(Feedforward):

	_product_base = MLP
	_linear_builder = 'linear'
	_nonlin_builder = 'nonlin'
	_norm_builder = 'norm'
	_dropout_builder = 'dropout'

	nonlin = hparam('elu', space=get_builder(_nonlin_builder).get_hparam('ident').space)
	out_nonlin = hparam(None, space=get_builder(_nonlin_builder).get_hparam('ident').space)

	norm = hparam(None, space=get_builder(_norm_builder).get_hparam('ident').space)
	out_norm = hparam(None, space=get_builder(_norm_builder).get_hparam('ident').space)

	hidden = hparam(())

	bias = hparam(True, space=spaces.Binary())
	out_bias = hparam(True, space=spaces.Binary())

	dropout = hparam(0., space=spaces.Bound(0., 1.))


	@agnostic
	def _build_block(self, builder, *, din=None, dout=None, **kwargs):
		layers = super()._build_block(self._linear_builder, width=builder, din=din, dout=dout,
		                              bias=self.out_bias if builder is None else self.bias, **kwargs)
		din, dout = layers[0].din, layers[0].dout

		if builder is None:
			if self.out_norm is not None:
				layers.extend(super()._build_block(self._norm_builder, ident=self.out_norm, width=dout.width))
			if self.out_nonlin is not None:
				layers.extend(super()._build_block(self._nonlin_builder, ident=self.out_nonlin))

		else:
			if self.norm is not None:
				layers.extend(super()._build_block(self._norm_builder, ident=self.norm, width=dout.width))
			if self.nonlin is not None:
				layers.extend(super()._build_block(self._nonlin_builder, ident=self.nonlin))
			if self.dropout > 0:
				layers.extend(super()._build_block(self._dropout_builder, p=self.dropout))

		return layers, din, dout


	@agnostic
	def build(self, layer_builders=None, din=None, dout=None, **kwargs):
		if layer_builders is None:
			layer_builders = [*self.hidden, None]
		return super().build(layer_builders, din=din, dout=dout, **kwargs)







# class NormalizationBuilder(reg.BranchBuilder, branch='norm', default_ident='batch', products={
# 							'batch1d': nn.BatchNorm1d,
# 							'batch2d': nn.BatchNorm2d,
# 							'batch3d': nn.BatchNorm3d,
# 							'instance1d': nn.InstanceNorm1d,
# 							'instance2d': nn.InstanceNorm2d,
# 							'instance3d': nn.InstanceNorm3d,
# 							'group': nn.GroupNorm,
#                          }):
#
# 	width = hparam(None)
# 	spatial_dims = hparam(1, space=[1, 2, 3])
#
# 	num_groups = hparam(None)
#
# 	momentum = hparam(0.1, space=spaces.Bound(0.01, 0.99))
# 	eps = hparam(1e-5)
# 	affine = hparam(True, space=spaces.Binary())
#
# 	@agnostic
# 	def product(self, ident, spatial_dims=1, **kwargs):
# 		if ident in {'batch', 'instance'}:
# 			ident = f'{ident}{spatial_dims}d'
# 		return super().product(ident, **kwargs)
#
# 	@agnostic
# 	def build(self, ident, width, spatial_dims=1, num_groups=None, momentum=0.1,
# 	           eps=1e-5, affine=True, **kwargs):
#
# 		product = self.product(ident=ident, spatial_dims=spatial_dims, **kwargs)
#
# 		if issubclass(product, nn.GroupNorm):
# 			if num_groups is None:
# 				num_groups = 8 if width >= 16 else 4
# 			return product(num_groups, width, eps=eps, affine=affine)
# 		return product(width, momentum=momentum, eps=eps, affine=affine, **kwargs)






#####

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














