import torch
import torch.nn as nn
from torch.distributions import Normal as NormalDistribution
# from .. import framework as fm
from ..op import framework as fm

import omnifig as fig

from .layers import make_MLP

#################
# region General
#################


@fig.Component('mlp')
class MLP(nn.Sequential, fm.FunctionBase):
	def __init__(self, A):
		kwargs = {
			'din': A.pull('input_dim', '<>din'),
			'dout': A.pull('output_dim', '<>dout'),
			'hidden': A.pull('hidden_dims', '<>hidden_fc', '<>hidden', []),
			'nonlin': A.pull('nonlin', 'elu'),
			'output_nonlin': A.pull('output_nonlin', '<>out_nonlin', None),
			'logify_in': A.pull('logify', False),
			'unlogify_out': A.pull('unlogify', False),
			'bias': A.pull('bias', True),
			'output_bias': A.pull('output_bias', '<>out_bias', None),
		}

		net = make_MLP(**kwargs)
		super().__init__(*net)
		self.din, self.dout = net.din, net.dout

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
class MultiLayer(fm.FunctionBase):

	def __init__(self, A=None, layers=None, **kwargs):

		super().__init__(A, **kwargs)

		if layers is None:
			layers = self._create_layers(A)

		self.din, self.dout = layers[0].din, layers[-1].dout

		self.layers = nn.ModuleList(layers)

	def _create_layers(self, A):

		din = A.pull('final_din', '<>din', None)
		dout = A.pull('final_dout', '<>dout', None)

		assert din is not None or dout is not None, 'need some input info'

		in_order = A.pull('in_order', din is not None)
		force_iter = A.pull('force_iter', True)

		create_layers = A.pull('layers', as_iter=force_iter)
		# create_layers = deepcopy(create_layers)
		create_layers.set_auto_pull(False)

		self._current = din if in_order else dout
		self._req_key = 'din' if in_order else 'dout'
		self._find_key = 'dout' if in_order else 'din'
		end = dout if in_order else din

		pre, post = ('first', 'last') if in_order else ('last', 'first')

		pre_layer = self._create_layer(A.sub(pre)) if pre in A else None

		mid = [self._create_layer(layer) for layer in create_layers]

		post_layer = None
		if end is not None or post in A:
			if post not in A:
				A.push('output_nonlin', None)
			mytype = A.push((post, '_type'), 'dense-layer', silent=True, overwrite=False)
			if end is not None:
				A.push((post, self._find_key), end, silent=True)
			if mytype == 'dense-layer' and in_order:
				A.push((post, 'nonlin'), '<>output_nonlin', silent=False, overwrite=False)
			post_layer = self._create_layer(A.sub(post), empty_find=end is None)

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
# region Behavior
#################

@fig.AutoModifier('normal')
class Normal(fm.FunctionBase):
	def __init__(self, A):
		super().__init__(A)

		dout = self.dout
		self.full_dout = dout

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

				out = nn.Linear(chn, chn*2)

				# A.push('out-layer._type', 'dense-layer', silent=True)
				# A.push('')

			else:
				chn = dout[0]

				out = nn.Conv2d(chn, chn*2, kernel_size=1)

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


class OldNormal(fm.FunctionBase):
	'''
	This is a modifier (basically mixin) to turn the parent's output of forward() to a normal distribution.

	'''

	def __init__(self, A, latent_dim=None):
		if latent_dim is None:
			dout = A.pull('latent_dim', '<>dout')

		if isinstance(dout, tuple):
			cut, *rest = dout
			full_dout = cut*2, *rest
		else:
			cut = dout
			full_dout = dout*2

		_dout, _latent_dim = A.pull('dout', None, silent=True), A.pull('latent_dim', None, silent=True)
		A.push('dout', full_dout, silent=True)
		A.push('latent_dim', full_dout, silent=True) # temporarily change

		min_log_std = A.pull('min_log_std', None)

		super().__init__(A)

		# reset config to correct terms
		if _dout is not None:
			A.push('dout', _dout, silent=True)
		if _latent_dim is not None:
			A.push('latent_dim', _latent_dim, silent=True)
		self.latent_dim = dout
		self.dout = dout

		self.cut = cut
		self.full_dout = full_dout

		self.min_log_std = min_log_std

	def forward(self, *args, **kwargs):

		q = super().forward(*args, **kwargs)

		mu, logsigma = q.narrow(1, 0, self.cut), q.narrow(1, self.cut, self.cut)

		if self.min_log_std is not None:
			logsigma = logsigma.clamp(min=self.min_log_std)

		return NormalDistribution(loc=mu, scale=logsigma.exp())

# endregion
#################








