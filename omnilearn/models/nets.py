import torch
import torch.nn as nn
from torch.distributions import Normal as NormalDistribution
# from .. import framework as fm
from ..op import framework as fm

from omnibelt import InitWall, unspecified_argument
import omnifig as fig

from .layers import make_MLP

#################
# region General
#################

class MLPBase(fm.FunctionBase, InitWall, nn.Sequential):
	def __init__(self, din, dout, hidden=None, initializer=None,
	             nonlin='elu', output_nonlin=None,
	             bias=True, output_bias=None, make_kwargs={}, _req_args=(), **kwargs):

		assert len(_req_args) == 0, str(_req_args)

		net = make_MLP(din=din, dout=dout, hidden=hidden, initializer=initializer,
		               nonlin=nonlin, output_nonlin=output_nonlin,
		               bias=bias, output_bias=output_bias, **make_kwargs)
		super().__init__(din=net.din, dout=net.dout, _req_args=tuple(net), **kwargs)

@fig.Component('mlp')
class MLP(fm.Function, MLPBase):
	def __init__(self, A, din=unspecified_argument, dout=unspecified_argument,
	             hidden=unspecified_argument, initializer=unspecified_argument,
	             nonlin=unspecified_argument, output_nonlin=unspecified_argument,
	             bias=unspecified_argument, output_bias=unspecified_argument,
	             make_kwargs=unspecified_argument, **kwargs):

		# region fill in args

		if din is unspecified_argument:
			din = A.pull('_din', '<>din')
		if dout is unspecified_argument:
			dout = A.pull('_dout', '<>dout')
		if hidden is unspecified_argument:
			hidden = A.pull('hidden_dims', '<>hidden_fc', '<>hidden', None)
		if initializer is unspecified_argument:
			initializer = A.pull('initializer', None)
		if nonlin is unspecified_argument:
			nonlin = A.pull('nonlin', 'elu')
		if output_nonlin is unspecified_argument:
			output_nonlin = A.pull('output_nonlin', '<>out_nonlin', None)
		if bias is unspecified_argument:
			bias = A.pull('bias', True)
		if output_bias is unspecified_argument:
			output_bias = A.pull('output_bias', '<>out_bias', None)
		if make_kwargs is unspecified_argument:
			make_kwargs = A.pull('make_kwargs', {})

		# endregion

		super().__init__(A, din=din, dout=dout,
		                 hidden=hidden, initializer=initializer,
		                 nonlin=nonlin, output_nonlin=output_nonlin,
		                 bias=bias, output_bias=output_bias,
		                 make_kwargs=make_kwargs, **kwargs)

@fig.Component('multihead')
class Multihead(fm.FunctionBase): # currently, the input dim for each head must be the same (but output can be different) TODO: generalize
	def __init__(self, A):
		
		din = A.pull('din')
		pin, N = din
		
		n_first = A.pull('n_first', False) # to change the "pin" and "N" dimensions
		if n_first:
			raise NotImplementedError
		
		merge = A.pull('merge', 'concat')
		if merge != 'concat':
			raise NotImplementedError
		
		A.push('heads.din', pin, silent=True)
		create_head = A.pull('heads')
		
		head_douts = A.pull('head_douts', None)
		try:
			len(head_douts)
		except TypeError:
			idouts = iter([head_douts]*len(create_head))
		else:
			idouts = iter(head_douts)
		
		heads = []
		
		nxt = create_head.view()
		if head_douts is not None:
			nxt.push('dout', next(idouts), silent=True)
		for head in create_head:
			heads.append(head)
			try:
				nxt = create_head.view()
			except StopIteration:
				break
			else:
				if head_douts is not None:
					nxt.push('dout', next(idouts), silent=True)
		
		assert N == len(heads), f'{N} vs {len(heads)}'
		
		douts = [head.dout for head in heads]
		
		if merge == 'concat':
			dout = sum(douts)
		
		super().__init__(din, dout)
		
		self.heads = nn.ModuleList(heads)
		
		self.merge = merge
		
	def forward(self, xs):
		
		xs = xs.permute(2,0,1)
		
		ys = [head(x) for head, x in zip(self.heads, xs)]
		
		return torch.cat(ys, dim=1)

@fig.Component('multilayer') # used for CNNs
class MultiLayer(fm.Function):

	def __init__(self, A, layers=None, din=None, dout=None, in_order=None, **kwargs):

		if layers is None:
			layers = self._create_layers(A, din=din, dout=dout, in_order=in_order)

		super().__init__(A, din=layers[0].din, dout=layers[-1].dout, **kwargs)

		self.layers = nn.ModuleList(layers)

	def _create_layers(self, A, din=None, dout=None, in_order=None):

		if din is None:
			din = A.pull('_din', '<>din', None)
		if dout is None:
			dout = A.pull('_dout', '<>dout', None)

		assert din is not None or dout is not None, 'need some input info'

		if in_order is None:
			in_order = A.pull('in_order', din is not None)

		self._current = din if in_order else dout
		self._req_key = 'din' if in_order else 'dout'
		self._find_key = 'dout' if in_order else 'din'
		end = dout if in_order else din

		pre, post = ('first', 'last') if in_order else ('last', 'first')

		pre_layer_info = A.pull(pre, None, raw=True)
		pre_layer = None if pre_layer_info is None else self._create_layer(pre_layer_info)

		mid = []
		create_layers = A.pull('layers', None, as_iter=True)
		if create_layers is not None:
			create_layers.set_auto_pull(False)  # prevents layer from being created before din/dout info is updated
			create_layers.set_reversed(not in_order)
	
			mid = [self._create_layer(layer) for layer in create_layers]

		post_layer = None
		post_layer_info = A.pull(post, None, raw=True)
		if end is not None or post_layer_info is not None:
			if post_layer_info is None:
				A.push('output_nonlin', None)
				post_layer_info = A.sub(post)
			mytype = post_layer_info.push('_type', 'dense-layer', silent=True, overwrite=False)
			if end is not None:
				post_layer_info.push(self._find_key, end, silent=True)
			if mytype == 'dense-layer' and in_order:
				post_layer_info.push('nonlin', '<>output_nonlin', silent=False, overwrite=False)
			post_layer = self._create_layer(post_layer_info, empty_find=end is None)

		layers = []
		if pre_layer is not None:
			layers.append(pre_layer)
		layers.extend(mid)
		if post_layer is not None:
			layers.append(post_layer)
		return layers if in_order else layers[::-1]

	def _create_layer(self, info, empty_find=True):
		current = self._current
		info.push(self._req_key, current, silent=True, overwrite=True)
		if empty_find:
			info.push(self._find_key, None, silent=True, overwrite=True)
		layer = info.pull_self()
		# c, n = self._current, getattr(layer, self._find_key)
		self._current = getattr(layer, self._find_key)
		return layer

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = layer(x)
		return x

# endregion
#################








