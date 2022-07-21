from omnibelt import agnosticmethod, unspecified_argument
import numpy as np
from torch import nn

from omnidata.framework import hparam, Function, Model, spaces



class Reshaper(nn.Module): # by default flattens

	def __init__(self, dout=(-1,)):
		super().__init__()

		self.dout = dout


	def extra_repr(self):
		return f'out={self.dout}'


	def forward(self, x):
		B = x.size(0)
		return x.view(B, *self.dout)



class MLP(Model, Function, nn.Sequential):
	# def __init__(self, layers, din=None, dout=None, **kwargs):
	# 	super().__init__(layers, din=din, dout=dout, **kwargs)

	nonlin = hparam('elu', space=['prelu', 'lrelu', 'relu', 'tanh',
	                              'softplus', 'sigmoid', 'elu', 'selu'])
	nonlin_inplace = hparam(True)
	nonlin_kwargs = hparam({})

	norm = hparam(None, space=['batch', 'instance'])
	norm_kwargs = hparam({})

	bias = hparam(True)

	out_nonlin = hparam(None)
	out_norm = hparam(None)
	out_bias = hparam(True)

	hidden = hparam(())


	class Builder(Model.Builder):
		@agnosticmethod
		def _expand_dim(self, dim):
			if isinstance(dim, int):
				dim = [dim]
			if isinstance(dim, (list, tuple)):
				return dim
			if isinstance(dim, spaces.Dim):
				return dim.expanded_shape
			raise NotImplementedError(dim)


		def _create_nonlin(self, ident=None, inplace=unspecified_argument, **_kwargs):
			# if ident is unspecified_argument:
			# 	ident = self.nonlin
			if inplace is unspecified_argument:
				inplace = self.nonlin_inplace
			kwargs = self.nonlin_kwargs.copy()
			kwargs.update(_kwargs)

			if ident is None:
				return None

			if ident == 'prelu':
				return nn.PReLU(**kwargs)
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
			elif ident == 'elu':
				return nn.ELU(inplace=inplace, **kwargs)
			elif ident == 'selu':
				return nn.SELU(inplace=inplace)
			raise self.UnknownNonlin(ident)


		class UnknownNonlin(NotImplementedError):
			pass


		def _create_norm(self, norm, width, **_kwargs):
			# if norm is unspecified_argument:
			# 	norm = self.norm
			# if width is unspecified_argument:
			# 	width = self.dout.expanded_len
			kwargs = self.norm_kwargs.copy()
			kwargs.update(_kwargs)

			if norm is None:
				return None

			if norm == 'batch':
				return nn.BatchNorm1d(width, **kwargs)
			if norm == 'instance':
				return nn.InstanceNorm1d(width, **kwargs)
			raise self.UnknownNorm


		class UnknownNorm(NotImplementedError):
			pass


		@agnosticmethod
		def _build_layer(self, din, dout, nonlin=unspecified_argument, norm=unspecified_argument,
		                 bias=unspecified_argument):

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

			layers.append(nn.Linear(in_width, out_width, bias=bias))

			nonlin = self._create_nonlin(nonlin)
			if nonlin is not None:
				layers.append(nonlin)

			norm = self._create_norm(norm, width=out_width)
			if norm is not None:
				layers.append(norm)

			if len(out_shape) > 1:
				layers.append(Reshaper(out_shape))
			return dout, layers


		@agnosticmethod
		def _build_outlayer(self, din, dout, nonlin=unspecified_argument, norm=unspecified_argument,
		                    bias=unspecified_argument):
			if nonlin is unspecified_argument:
				nonlin = self.out_nonlin
			if norm is unspecified_argument:
				norm = self.out_norm
			if bias is unspecified_argument:
				bias = self.out_bias
			return self._build_layer(din, dout, nonlin=nonlin, norm=norm, bias=bias)[1]


		@agnosticmethod
		def _build_layers(self, din, dout, hidden=()):
			layers = []
			start_dim, end_dim = din, din
			for end_dim in hidden:
				start_dim, new_layers = self._build_layer(start_dim, end_dim)
				layers.extend(new_layers)
			layers.extend(self._build_outlayer(end_dim, dout))
			return layers


		def build(self, kwargs=None):
			if kwargs is None:
				kwargs = dict(layers=self._build_layers(din=self.din, dout=self.dout, hidden=self.hidden),
			                     din=self.din, dout=self.dout)
			return super().build(kwargs)














