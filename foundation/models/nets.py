
import numpy as np
import torch
import torch.nn as nn
from itertools import zip_longest
import torch.nn.functional as F
from torch.distributions import Normal as NormalDistribution
from .. import framework as fm

import omnifig as fig

from .atom import *
from .layers import *

@fig.Component('mlp')
class MLP(fm.Model):
	def __init__(self, A):
		kwargs = {
			'input_dim': A.pull('input_dim', '<>din'),
			'output_dim': A.pull('output_dim', '<>dout'),
			'hidden_dims': A.pull('hidden_dims', '<>hidden_fc', []),
			'nonlin': A.pull('nonlin', 'prelu'),
			'output_nonlin': A.pull('output_nonlin', None),
		}

		net = make_MLP(**kwargs)
		super().__init__(kwargs['input_dim'], kwargs['output_dim'])

		self.net = net

	def __iter__(self):
		return iter(self.net)

	def __len__(self):
		return len(self.net)

	def __getitem__(self, item):
		return self.net[item]

	def forward(self, x):
		return self.net(x)

@fig.Component('multihead')
class Multihead(fm.Model): # currently, the input dim for each head must be the same (but output can be different) TODO: generalize
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


@fig.Component('double-enc')
class Double_Encoder(fm.Encodable, fm.Schedulable, fm.Model):
	def __init__(self, A):
		
		in_shape = A.pull('in_shape', '<>din')
		# latent_dim = A.pull('latent_dim', '<>dout')
		
		channels = A.pull('channels')
		assert isinstance(in_shape, tuple), 'input must be an image'

		factors = A.pull('factors', 2)
		try:
			len(factors)
		except TypeError:
			factors = [factors]
		if len(factors) != len(channels):
			factors = factors*len(channels)
		total_factor = int(np.product(factors))

		internal_channels = A.pull('internal_channels', [None] * len(channels))
		try:
			len(internal_channels)
		except TypeError:
			internal_channels = [internal_channels]
		if len(internal_channels) != len(channels):
			internal_channels = internal_channels * len(channels)

		squeeze = A.pull('squeeze', [False] * len(channels))
		try:
			len(squeeze)
		except TypeError:
			squeeze = [squeeze]
		if len(squeeze) != len(channels):
			squeeze = squeeze * len(channels)

		output_nonlin = A.pull('output_nonlin', None)

		din = in_shape
		in_chn, *in_size = in_shape
		out_chn = channels[-1]
		if len(in_size):
			in_H, in_W = in_size
			out_H, out_W = in_H//total_factor, in_W//total_factor
			assert out_H > 0 and out_W > 0, 'out image not large enough: {} {}'.format(out_H, out_W)
			dout = out_chn, out_H, out_W
		else:
			dout = out_chn,

		latent_dim = A.pull('latent_dim', '<>dout')
		if 'tail' in A:
			set_nonlin = True
			A.tail.din = dout
			A.tail.dout = latent_dim
			tail = A.pull('tail')
			assert tail.dout == latent_dim, \
				'Tail not producing the right output: {} vs {}'.format(tail.dout, latent_dim)
		elif latent_dim != dout:
			set_nonlin = True
			assert len(dout) == 3, 'Must specify image size to transform to {}'.format(latent_dim)
			tail = make_MLP(input_dim=dout, output_dim=latent_dim, output_nonlin=output_nonlin)
		else:
			set_nonlin = False
			tail = None

		dout = latent_dim

		chns = (in_chn,) + channels
		layers = self._create_layers(chns, iter(factors), iter(internal_channels), iter(squeeze), A, set_nonlin)

		super().__init__(din, dout)

		self.layers = layers

		self.tail = tail

		# self.set_optim(A)
		# self.set_scheduler(A)

	def _create_layers(self, chns, factors, internal_channels, squeeze, A, set_nonlin):

		nonlin = A.pull('nonlin', 'elu')
		output_nonlin = A.pull('output_nonlin', None)
		output_norm_type = A.pull('output_norm', None)

		down_type = A.pull('down', 'max')
		norm_type = A.pull('norm', None)
		residual = A.pull('residual', False)

		last_chn = chns[-2:]
		chns = chns[:-1]

		layers = []
		for ichn, ochn in zip(chns, chns[1:]):
			layers.append(
				layerslib.DoubleConvLayer(in_channels=ichn, out_channels=ochn, factor=next(factors),
				                          down_type=down_type, norm_type=norm_type,
				                          nonlin=nonlin, output_nonlin=nonlin,
				                          internal_channels=next(internal_channels), squeeze=next(squeeze),
				                          residual=residual,
				                          )
			)
		layers.append(
			layerslib.DoubleConvLayer(in_channels=last_chn[0], out_channels=last_chn[1], factor=next(factors),
			                          down_type=down_type, norm_type=norm_type if set_nonlin else output_norm_type,
			                          nonlin=nonlin, output_nonlin=nonlin if set_nonlin else output_nonlin,
			                          internal_channels=next(internal_channels), squeeze=next(squeeze),
			                          residual=residual,
			                          )
		)
		return nn.ModuleList(layers)

	def encode(self, x):
		return self(x)
	
	def forward(self, x):

		q = x
		for l in self.layers:
			q = l(q)

		if self.tail is not None:
			q = self.tail(q)

		return q

@fig.Component('double-dec')
class Double_Decoder(fm.Decodable, fm.Schedulable, fm.Model):
	def __init__(self, A):

		out_shape = A.pull('out_shape', '<>dout')
		# latent_dim = A.pull('latent_dim', '<>dout')

		channels = A.pull('channels')
		assert isinstance(out_shape, tuple), 'input must be an image'

		factors = A.pull('factors', 2)
		try:
			len(factors)
		except TypeError:
			factors = [factors]
		if len(factors) != len(channels):
			factors = factors * len(channels)
		total_factor = int(np.product(factors))

		internal_channels = A.pull('internal_channels', [None] * len(channels))
		try:
			len(internal_channels)
		except TypeError:
			internal_channels = [internal_channels]
		if len(internal_channels) != len(channels):
			internal_channels = internal_channels * len(channels)

		squeeze = A.pull('squeeze', [False] * len(channels))
		try:
			len(squeeze)
		except TypeError:
			squeeze = [squeeze]
		if len(squeeze) != len(channels):
			squeeze = squeeze * len(channels)

		nonlin = A.pull('nonlin', 'elu')

		dout = out_shape
		out_chn, *out_size = out_shape
		in_chn = channels[0]
		if len(out_size):
			out_H, out_W = out_size
			in_H, in_W = out_H // total_factor, out_W // total_factor
			assert in_H > 0 and in_W > 0, 'out image not large enough: {} {}'.format(in_H, in_W)
			din = in_chn, in_H, in_W
		else:
			din = in_chn,

		latent_dim = A.pull('latent_dim', '<>din')
		if 'head' in A:
			A.head.din = latent_dim
			A.head.dout = din
			A.head.output_nonlin = nonlin
			head = A.pull('head')
		elif latent_dim != dout:
			assert len(dout) == 3, 'Must specify image size to transform to {}'.format(latent_dim)
			head = make_MLP(input_dim=latent_dim, output_dim=din, output_nonlin=nonlin)
		else:
			head = None

		din = latent_dim

		chns = channels + (out_chn,)
		layers = self._create_layers(chns, iter(factors), iter(internal_channels), iter(squeeze), A)

		super().__init__(din, dout)

		self.head = head
		self.layers = layers

		# self.set_optim(A)
		# self.set_scheduler(A)

	def _create_layers(self, chns, factors, internal_channels, squeeze, A):

		nonlin = A.pull('nonlin', 'elu')
		output_nonlin = A.pull('output_nonlin', None)
		output_norm_type = A.pull('output_norm', None)

		up_type = A.pull('up', 'bilinear')
		norm_type = A.pull('norm', None)
		residual = A.pull('residual', False)

		last_chn = chns[-2:]
		chns = chns[:-1]

		layers = []

		# i_factors, i_internal_channels, i_squeeze =
		for ichn, ochn in zip(chns, chns[1:]):
			layers.append(
				layerslib.DoubleDeconvLayer(in_channels=ichn, out_channels=ochn, factor=next(factors),
				                            up_type=up_type, norm=norm_type,
				                            nonlin=nonlin, output_nonlin=nonlin,
				                            internal_channels=next(internal_channels), squeeze=next(squeeze),
				                            residual=residual,
				                            )
			)
		layers.append(
			layerslib.DoubleDeconvLayer(in_channels=last_chn[0], out_channels=last_chn[1], factor=next(factors),
			                            up_type=up_type, norm=output_norm_type,
			                            nonlin=nonlin, output_nonlin=output_nonlin,
			                            internal_channels=next(internal_channels), squeeze=next(squeeze),
			                            residual=residual,
			                            )
		)

		return nn.ModuleList(layers)

	def decode(self, q):
		return self(q)

	def forward(self, q):

		x = q

		if self.head is not None:
			x = self.head(x)

		for l in self.layers:
			x = l(x)

		return x

class Normal_Distrib_Model(fm.Model): # means calling forward() will output a normal distribution
	pass

@fig.AutoModifier('normal')
class Normal(Normal_Distrib_Model):
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

		_dout, _latent_dim = A.dout, A.latent_dim
		A.dout = full_dout
		A.latent_dim = full_dout # temporarily change

		min_log_std = A.pull('min_log_std', None)

		super().__init__(A)

		# reset config to correct terms
		A.dout, A.latent_dim = _dout, _latent_dim
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

@fig.AutoComponent('conv')
class Conv_Encoder(fm.Encodable, fm.Model):

	def __init__(self, in_shape, latent_dim=None, feature_dim=None,
				 nonlin='prelu', output_nonlin=None, residual=False,
				 channels=[], kernels=3, strides=1, factors=2, down='max',
				 norm='instance', output_norm=None,
				 fc_hidden=[]):
		
		self.in_shape = in_shape

		cshapes, csets = plan_conv(self.in_shape, channels=channels, kernels=kernels, factors=factors, strides=strides)

		conv_layers = build_conv_layers(csets, factors=factors, pool_type=down, norm_type=norm,
										out_norm_type=(output_norm if latent_dim is None else norm),
										nonlin=nonlin, residual=residual,
										out_nonlin=(output_nonlin if latent_dim is None else nonlin))

		out_shape = cshapes[-1]

		super().__init__(in_shape, out_shape if latent_dim is None else latent_dim)

		self.conv = nn.Sequential(*conv_layers)
		self._conv_shapes = cshapes

		self.conv_shape = out_shape
		if feature_dim is None:
			feature_dim = self.conv_shape
		self.feature_dim = feature_dim
		self.latent_dim = latent_dim if latent_dim is not None else self.feature_dim

		self.fc = None
		if latent_dim is not None:
			self.fc = make_MLP(self.feature_dim, self.latent_dim,
							   hidden_dims=fc_hidden, nonlin=nonlin,
							   output_nonlin=output_nonlin)

	def transform_conv_features(self, c):
		if self.fc is None:
			return c
		return self.fc(c)

	def encode(self, x):
		return self(x)

	def forward(self, x):
		c = self.conv(x)
		return self.transform_conv_features(c)

@fig.AutoComponent('deconv')
class Conv_Decoder(fm.Decodable, fm.Model):
	
	def __init__(self, out_shape, latent_dim=None, nonlin='prelu', output_nonlin=None,
	             channels=[], kernels=[], factors=[], strides=[], up='deconv',
	             norm='instance', output_norm=None, residual=False,
	             fc_hidden=[]):
		
		self.out_shape = out_shape
		
		dshapes, dsets = plan_deconv(self.out_shape, channels=channels, kernels=kernels, factors=factors,
		                             strides=strides if up == 'deconv' else 1)
		
		# dsizes = [(s if s[-2:] != ps[-2:] else None) for ps, s in zip(dshapes, dshapes[1:])]
		
		deconv_layers = build_deconv_layers(dsets, sizes=dshapes[1:], nonlin=nonlin, out_nonlin=output_nonlin,
		                                    up_type=up, norm_type=norm, factors=factors, residual=residual,
		                                    out_norm_type=output_norm)
		
		super().__init__(dshapes[0] if latent_dim is None else latent_dim, out_shape)
		
		self.deconv_shape = dshapes[0]
		self.latent_dim = latent_dim if latent_dim is not None else int(np.product(self.deconv_shape))
		
		self.fc = None
		if latent_dim is not None:
			self.fc = make_MLP(self.latent_dim, self.deconv_shape,
			                   hidden_dims=fc_hidden, nonlin=nonlin, output_nonlin=nonlin)
		
		self.deconv = nn.Sequential(*deconv_layers)
	
	def forward(self, q):
		c = self.extract_conv_features(q)
		return self.deconv(c)
	
	def extract_conv_features(self, q):
		if self.fc is not None:
			return self.fc(q)
		return q
		
	def decode(self, q):
		return self(q)


# TODO: update
class Rec_Encoder(Conv_Encoder): # fc before and after recurrence
	def __init__(self, in_shape, rec_dim,
	             
	             nonlin='prelu', before_fc=[], after_fc=[], out_dim=None,
	             output_nonlin=None, latent_dim=None,
	             
	             rec_type='lstm', rec_layers=1, auto_reset=True,
	             batch_first=False,
	             
	             **kwargs):
		
		super().__init__(in_shape, latent_dim=None, **kwargs)

		
		self.tfm = None
		if latent_dim is None and len(before_fc):
			latent_dim = before_fc[-1]
			before_fc = before_fc[:-1]
		if latent_dim is not None:
			self.tfm = make_MLP(self.feature_dim, latent_dim, hidden_dims=before_fc,
			                       nonlin=nonlin, output_nonlin=nonlin)
		else:
			latent_dim = self.conv_dim
			
		self.latent_dim = latent_dim # input to rec
		
		if out_dim is None:
			if len(after_fc):
				out_dim = after_fc[-1]
				after_fc = after_fc[:-1]
			else:
				out_dim = rec_dim
				rec_dim = None
		if len(after_fc):
			rec_out_dim = rec_dim
			rec_hidden_dim = None
		else:
			rec_out_dim = out_dim
			rec_hidden_dim = rec_dim
			after_fc = None

		self.rec = Recurrence(latent_dim, output_dim=rec_out_dim, hidden_dim=rec_hidden_dim,
		                      auto_reset=auto_reset, batch_first=batch_first,
		                      output_nonlin=(output_nonlin if after_fc is None else None),
		                      rec_type=rec_type, num_layers=rec_layers)
		
		self.dec = None
		if after_fc is not None:
			self.dec = make_MLP(rec_out_dim, out_dim, hidden_dims=after_fc,
			                    nonlin=nonlin, output_nonlin=output_nonlin)
			
		
		self.dout = out_dim

	def reset(self):
		super().reset()
		self.rec.reset()
		
	def forward(self, xs):
		if xs.ndimension() == 4:
			xs = xs.unsqueeze(int(self.rec.batch_first))
			
		K, B, C, H, W = xs.size()
		xs = xs.view(K*B,C,H,W)
		
		cs = self.conv(xs)
		
		if self.tfm is not None:
			cs = self.tfm(cs.view(K*B, self.conv_dim))
			
		qs = self.rec(cs.view(K,B,self.latent_dim))
		
		if self.dec is not None:
			qs = self.dec(qs)
			
		return qs
	











