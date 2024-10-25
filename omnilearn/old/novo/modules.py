from typing import Iterable, Any, Optional, Union, Callable, Tuple, Dict, List
import numpy as np
from torch import nn
import torch.optim as O
from omnibelt import unspecified_argument, agnostic

import omnifig as fig

from omniplex import Function, spaces, SimpleFunction

from .base import hparam, inherit_hparams, submodule, submachine, material, space, indicator, machine, \
	Structured, Builder, get_builder
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


	# @space('input')
	# def input_space(self, output):
	# 	return output
	# @space('output')
	# def output_space(self, input):
	# 	return input


	def _build_kwargs(self, product, ident, **kwargs):
		kwargs = super()._build_kwargs(product, ident, **kwargs)
		if issubclass(product, (nn.ELU, nn.ReLU, nn.SELU)) and 'inplace' not in kwargs:
			kwargs['inplace'] = self.inplace
		return kwargs



class NormalizationBuilder(reg.BranchBuilder, branch='norm', default_ident='batch', products={
							'batch1d': nn.BatchNorm1d,
							'batch2d': nn.BatchNorm2d,
							'batch3d': nn.BatchNorm3d,
							'instance1d': nn.InstanceNorm1d,
							'instance2d': nn.InstanceNorm2d,
							'instance3d': nn.InstanceNorm3d,
							'group': nn.GroupNorm,
                         }):
	@hparam(cache=False)
	def width(self):
		# try:
		# 	width = self.dout.width
		# except AttributeError:
		# 	try:
		# 		width = self.din.width
		# 	except AttributeError:
		# 		raise ValueError('Could not determine width without input or output space')
		# return width
		if self.din is not None:
			return self.din.width
		if self.dout is not None:
			return self.dout.width
		raise ValueError('Could not determine width without input or output space')

	@hparam(cache=False, space=[1,2,3])
	def spatial_dims(self):
		# try:
		# 	return len(self.din.shape)
		# except AttributeError:
		# 	try:
		# 		return len(self.dout.shape)
		# 	except AttributeError:
		# 		return 1
		if self.din is not None:
			return len(self.din.shape)
		if self.dout is not None:
			return len(self.dout.shape)
		return 1

	@hparam(cache=False)
	def num_groups(self):
		width = self.width
		if width > 16:
			return 8
		if width > 4:
			return 4
		if width > 2:
			return 2
		return 1

	momentum = hparam(0.1, space=spaces.Bound(0.01, 0.99))
	eps = hparam(1e-5)
	affine = hparam(True, space=spaces.Binary())


	din = space('input')
	dout = space('output')


	def product_base(self, ident, spatial_dims=None, **kwargs):
		if spatial_dims is None:
			spatial_dims = self.spatial_dims
		if ident in {'batch', 'instance'}:
			ident = f'{ident}{spatial_dims}d'
		return super().product_base(ident, **kwargs)


	def _build_kwargs(self, product, ident, **kwargs):
		kwargs = super()._build_kwargs(product, ident, **kwargs)
		if issubclass(product, nn.GroupNorm):
			kwargs['num_groups'] = self.num_groups
		return kwargs



class CrossEntropyLoss(SimpleFunction, nn.CrossEntropyLoss, output='loss', inputs=('input', 'target')):
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
			raise NotImplementedError(target_space)



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
	width = hparam(None)

	bias = hparam(True, space=spaces.Binary())

	din = space('input')
	dout = space('output')


	def __init__(self, width: int = None, *args, **kwargs):
		super().__init__(*args, width=width, **kwargs)


	def product_base(self, *args, **kwargs):
		return nn.Linear


	def validate(self, product):
		if isinstance(product, int):
			return self.build(product)
		return super().validate(product)


	def _build_kwargs(self, product, width=unspecified_argument, *,
	                  in_features=None, out_features=None, bias=None, **kwargs):
		kwargs = super()._build_kwargs(product, **kwargs)

		if bias is None:
			bias = self.bias
		kwargs['bias'] = bias

		if width is unspecified_argument:
			width = self.width

		if in_features is None:
			# try:
			# 	in_features = self.din.width
			# except AttributeError:
			# 	if self.width is None:
			# 		raise
			# 	in_features = self.width
			in_features = width if self.din is None else self.din.width
		kwargs['in_features'] = in_features

		if out_features is None:
			# try:
			# 	out_features = self.dout.width
			# except AttributeError:
			# 	if self.width is None:
			# 		raise
			# 	out_features = self.width
			out_features = width if self.dout is None else self.dout.width
		kwargs['out_features'] = out_features

		return kwargs



@reg.builder('dropout')
class BasicDropout(Builder): # TODO: enable multiple dims
	p = hparam(0.1, space=spaces.Bound(0, 1))


	def product_base(self, *args, **kwargs):
		return nn.Dropout


	def _build_kwargs(self, product, *, p=None, **kwargs):
		kwargs = super()._build_kwargs(product, **kwargs)
		if p is None:
			p = self.p
		kwargs['p'] = p
		return kwargs



class AbstractBlock:
	def block_layers(self):
		yield from ()



class DenseLayer(SimpleFunction, nn.Module, AbstractBlock, output='output', inputs=('input',)):
	linear = submodule(builder='linear')

	norm = submodule(None, builder='norm')
	nonlin = submodule('elu', builder='nonlin')
	dropout = submodule(None, builder='dropout')


	din = space('input')
	dout = space('output')


	def __init__(self, linear: int = None, **kwargs):
		super().__init__(linear=linear, **kwargs)


	def block_layers(self):
		if self.linear is not None:
			yield self.linear
		if self.norm is not None:
			yield self.norm
		if self.nonlin is not None:
			yield self.nonlin
		if self.dropout is not None:
			yield self.dropout


	def forward(self, x):
		x = self.linear(x)
		if self.norm is not None:
			x = self.norm(x)
		if self.nonlin is not None:
			x = self.nonlin(x)
		if self.dropout is not None:
			x = self.dropout(x)
		return x



class Sequential(SimpleFunction, nn.Sequential, AbstractBlock, output='output', inputs='input'):
	din = space('input')
	dout = space('output')


	def block_layers(self):
		yield from self



@reg.builder('sequential')
class SequentialBuilder(Builder):
	'''
	This is effectively an abstract builder - it does not build anything by itself, but rather you can inherit
	from it and make sure all the layers are passed in with the keyword `layers`.
	'''
	din = space('input')
	dout = space('output')


	def product_base(self, *args, **kwargs):
		return nn.Sequential


	def product_signatures(self, *args, **kwargs):
		yield self._Signature('output', inputs='input')


	def _build_kwargs(self, product, layers=(), **kwargs):
		kwargs = super()._build_kwargs(product, **kwargs)
		if None not in kwargs:
			kwargs[None] = layers
		return kwargs



class UnwrappedBlocks(SequentialBuilder):
	@staticmethod
	def _layers_from_block(block):
		if isinstance(block, AbstractBlock):
			yield from block.block_layers()
		elif type(block) in {nn.Sequential, nn.ModuleList}:
			yield from block
		else:
			yield block


	def _build_kwargs(self, product, layers=(), **kwargs):
		layers = [layer for block in layers for layer in self._layers_from_block(block)]
		return super()._build_kwargs(product, layers=tuple(layers), **kwargs)



@reg.builder('feedforward')
class FeedforwardBuilder(SequentialBuilder):
	def _block_spec(self, overrides):
		overrides = {k: v for k, v in overrides.items() if v is not None}
		if self.my_blueprint is not None:
			return self.my_blueprint.adapt(overrides) # TODO: implement adapt in spec
		spec = self._Spec()
		for k, v in overrides.items():
			spec.change_space_of(k, v)
		return spec


	def _build_block(self, builder, spec=None):
		# if spec is None:
		# 	return builder.build() #if value is unspecified_argument else builder.validate(value)
		# return builder.build_from_spec(spec) #\
		# 	# if value is unspecified_argument else builder.validate_from_spec(value, spec=spec)
		builder = get_builder(builder)
		if isinstance(builder, type):
			builder = builder(blueprint=spec)
		if isinstance(builder, Builder):
			return builder.build() if spec is None else builder.build_from_spec(spec)
		return builder


	def _build_blocks(self, block_builders):
		reverse = self.din is None

		in_gizmo, out_gizmo = ('output', 'input') if reverse else ('input', 'output')

		ends = [None] * len(block_builders)
		start, ends[-1] = (self.dout, self.din) if reverse else (self.din, self.dout)

		for builder, end in zip(reversed(block_builders) if reverse else block_builders, ends):
			spec = self._block_spec({in_gizmo: start, out_gizmo: end})
			block = self._build_block(builder, spec=spec)
			start = spec.space_of(out_gizmo)
			yield block


	def _build_kwargs(self, product, *, layers=None, block_builders=None, **kwargs):
		if layers is None and block_builders is not None and len(block_builders):
			layers = [block for block in self._build_blocks(block_builders)]
		return super()._build_kwargs(product, layers=layers, **kwargs)



class MLP(Structured, nn.Sequential):
	pass



@reg.builder('mlp')
class MLPBuilder(FeedforwardBuilder, UnwrappedBlocks):

	hidden = hparam(())

	nonlin = hparam('elu', space=get_builder('nonlin').get_hparam('ident').space)
	out_nonlin = hparam(None, space=get_builder('nonlin').get_hparam('ident').space)

	norm = hparam(None, space=get_builder('norm').get_hparam('ident').space)
	out_norm = hparam(None, space=get_builder('norm').get_hparam('ident').space)

	bias = hparam(True, space=spaces.Binary())
	out_bias = hparam(True, space=spaces.Binary())

	dropout = hparam(0., space=spaces.Bound(0., 1.))


	def _build_kwargs(self, product, *, block_builders=None, **kwargs):
		if block_builders is None:

			pass

		return super()._build_kwargs(product, layers=layers, **kwargs)


	def product_base(self, *args, **kwargs):
		return MLP



# class MLP(Sequential): # just for the name
# 	pass



# @reg.builder('mlp')
# class MLPBuilder(FeedforwardBuilder):
#
# 	_product_base = MLP
# 	_linear_builder = 'linear'
# 	_nonlin_builder = 'nonlin'
# 	_norm_builder = 'norm'
# 	_dropout_builder = 'dropout'
#
# 	nonlin = hparam('elu', space=get_builder(_nonlin_builder).get_hparam('ident').space)
# 	out_nonlin = hparam(None, space=get_builder(_nonlin_builder).get_hparam('ident').space)
#
# 	norm = hparam(None, space=get_builder(_norm_builder).get_hparam('ident').space)
# 	out_norm = hparam(None, space=get_builder(_norm_builder).get_hparam('ident').space)
#
# 	hidden = hparam(())
#
# 	bias = hparam(True, space=spaces.Binary())
# 	out_bias = hparam(True, space=spaces.Binary())
#
# 	dropout = hparam(0., space=spaces.Bound(0., 1.))
#
#
# 	@agnostic
# 	def _build_block(self, builder, *, din=None, dout=None, **kwargs):
# 		layers = super()._build_block(self._linear_builder, width=builder, din=din, dout=dout,
# 		                              bias=self.out_bias if builder is None else self.bias, **kwargs)
# 		din, dout = layers[0].din, layers[0].dout
#
# 		if builder is None:
# 			if self.out_norm is not None:
# 				layers.extend(super()._build_block(self._norm_builder, ident=self.out_norm, width=dout.width))
# 			if self.out_nonlin is not None:
# 				layers.extend(super()._build_block(self._nonlin_builder, ident=self.out_nonlin))
#
# 		else:
# 			if self.norm is not None:
# 				layers.extend(super()._build_block(self._norm_builder, ident=self.norm, width=dout.width))
# 			if self.nonlin is not None:
# 				layers.extend(super()._build_block(self._nonlin_builder, ident=self.nonlin))
# 			if self.dropout > 0:
# 				layers.extend(super()._build_block(self._dropout_builder, p=self.dropout))
#
# 		return layers, din, dout
#
#
# 	@agnostic
# 	def build(self, layer_builders=None, din=None, dout=None, **kwargs):
# 		if layer_builders is None:
# 			layer_builders = [*self.hidden, None]
# 		return super().build(layer_builders, din=din, dout=dout, **kwargs)







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














