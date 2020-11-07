import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

import omnifig as fig

from .. import framework as fm
from .. import util


#################
# region Functions
#################

def make_MLP(din, dout, hidden=None,
             initializer=None,
             nonlin='elu', output_nonlin=None,
             logify_in=False, unlogify_out=False,
             bias=True, output_bias=None):
	'''
	:param din: int
	:param dout: int
	:param hidden: ordered list of int - each element corresponds to a FC layer with that width (empty means network is not deep)
	:param nonlin: str - choose from options found in get_nonlinearity(), applied after each intermediate layer
	:param output_nonlin: str - nonlinearity to be applied after the last (output) layer
	:param logify_in: convert input to logified space
	:param unlogify_out: convert output back to linear space
	:return: an nn.Sequential instance with the corresponding layers
	'''

	if hidden is None:
		hidden = []

	if output_bias is None:
		output_bias = bias

	flatten = False
	reshape = None

	din = din
	dout = dout

	if isinstance(din, (tuple, list)):
		flatten = True
		din = int(np.product(din))
	if isinstance(dout, (tuple, list)):
		reshape = dout
		dout = int(np.product(dout))

	nonlins = [nonlin] * len(hidden) + [output_nonlin]
	biases = [bias] * len(hidden) + [output_bias]
	hidden = din, *hidden, dout

	layers = []
	if flatten:
		layers.append(nn.Flatten())

	if logify_in:
		layers.append(util.Logifier())
	for in_dim, out_dim, nonlin, bias in zip(hidden, hidden[1:], nonlins, biases):
		layer = nn.Linear(in_dim, out_dim, bias=bias)
		if initializer is not None:
			layer = initializer(layer, nonlin)
		layers.append(layer)
		if nonlin is not None:
			layers.append(util.get_nonlinearity(nonlin))

	if unlogify_out:
		layers.append(util.Unlogifier())
	if reshape is not None:
		layers.append(Reshaper(reshape))


	net = nn.Sequential(*layers)

	net.din, net.dout = din, dout
	return net

def batched_grouped_linear(points, weights, masks, biases=None): # equivalent to Ntfm
	'''
	Linear layer where the weights (and biases) are batched and grouped as well,
	so the output is a weighted sum of the outputs from each group

	Use case: points=point_cloud, weights=SE3_rotations, masks=segmentation_masks, biases=SE_translation

	points: (B, C, M)
	weights: (B, N, D, C)
	biases: (B, N, D)
	masks: (B, N, M) - sum across N == 1

	out: (B, D, M)
	'''

	features = weights @ points.unsqueeze(1)  # (B, N, D, M)

	#print(points.shape, weights.shape, masks.shape, biases.shape, features.shape)

	if biases is not None:
		features += biases#.unsqueeze(-1)

	out = features.mul(masks.unsqueeze(2)).sum(1)

	return out

fig.AutoComponent('flatten')(nn.Flatten)

@fig.AutoComponent('reshaper')
class Reshaper(nn.Module): # by default flattens

	def __init__(self, dout=(-1,)):
		super().__init__()

		self.dout = dout

	def extra_repr(self):
		return f'out={self.dout}'

	def forward(self, x):
		B = x.size(0)
		return x.view(B, *self.dout)

@fig.Modification('spec-norm')
def spec_norm_layer(layer, config):
	
	kwargs = dict(
		name = config.pull('weight_name', 'weight'),
		n_power_iterations = config.pull('n_power_iterations', 1),
		eps = config.pull('eps', 1e-12),
		dim = config.pull('dim', None),
	)
	
	return spectral_norm(layer, **kwargs)

# endregion
#################
# region Layers
#################

@fig.AutoComponent('recurrence')
class Recurrence(fm.Model):
	def __init__(self, din, dout, hidden_dim=None, num_layers=1, output_nonlin=None,
	             rec_type='gru', dropout=0., batch_first=False, bidirectional=False, auto_reset=True):
		super().__init__(din, dout)

		assert rec_type in {'gru', 'lstm', 'rnn'}

		self.rec_dim = hidden_dim
		if hidden_dim is None:
			self.rec_dim = dout

		rec_args = {'input_size': din, 'hidden_size': self.rec_dim, 'num_layers': num_layers,
					'dropout': dropout, 'batch_first': batch_first, 'bidirectional': bidirectional}
		if rec_type == 'rnn':
			self.rec = nn.RNN(**rec_args)
		elif rec_type == 'rnn-relu':
			self.rec = nn.RNN(nonlinearity='relu', **rec_args)
		elif rec_type == 'gru':
			self.rec = nn.GRU(**rec_args)
		elif rec_type == 'lstm':
			self.rec = nn.LSTM(**rec_args)

		self.out_layer = None
		self.out_nonlin = None
		if hidden_dim is not None:
			self.out_layer = make_MLP(self.rec_dim, dout, output_nonlin=output_nonlin)

		self.auto_reset = auto_reset
		self.reset()

	def reset(self, hidden=None):
		self.hidden = hidden

	def forward(self, seq, hidden=None, ret_hidden=False):  # input dims: seq x batch x input

		if self.auto_reset:
			self.reset(hidden)
		hidden = self.hidden if hidden is None else hidden

		out, self.hidden = self.rec(seq, hidden)  # if self.hidden is None else self.rec(seq, self.hidden)

		if self.out_layer is not None:
			out = self.out_layer(out)

		if ret_hidden:
			return out, self.hidden

		return out

@fig.AutoComponent('fourier-layer')
class Fourier_Layer(fm.Model): # TODO: generalize period to multiple din
	def __init__(self, din=1, dout=1, order=100, periods=None):
		super().__init__(din, dout)

		self.sin_coeff = nn.Linear(self.din * order, self.dout, bias=False)
		self.cos_coeff = nn.Linear(self.din * order, self.dout, bias=True)
		self.order = order

		if periods is not None:
			w = 2* np.pi / periods
			self.register_buffer('w', w.view(1, -1))
		else:
			w = nn.Parameter(torch.tensor([2 * np.pi] * din).view(1, -1), requires_grad=True)
			self.register_parameter('w', w)

		freqs = torch.arange(1, order + 1).view(1, 1, -1).float()  # 1 x 1 x O
		self.register_buffer('freqs', freqs)

	def forward(self, x):

		x = x.view(-1, self.din).mul(self.w).unsqueeze(-1)  # B x D x 1
		x = x.mul(self.freqs).view(-1, self.din * self.order)

		return self.sin_coeff(torch.sin(x)) + self.cos_coeff(torch.cos(x))

	def get_period(self):
		return 2 * np.pi / self.w

class PowerLinear(fm.Model):  # includes powers of input as features
	def __init__(self):
		raise NotImplementedError


@fig.AutoComponent('dense-layer')
class DenseLayer(fm.Model):
	def __init__(self, din, dout, bias=True, nonlin='elu'):
		super().__init__(din, dout)

		seq = []

		if isinstance(din, (tuple, list)):
			seq.append(nn.Flatten())
			din = int(np.product(din))

		_dout = int(np.product(dout)) if isinstance(dout, (tuple, list)) else dout

		seq.append(nn.Linear(din, _dout, bias=bias))

		nonlin = util.get_nonlinearity(nonlin)
		if nonlin is not None:
			seq.append(nonlin)

		if isinstance(dout, (tuple, list)):
			seq.append(Reshaper(dout))

		self.layer = nn.Sequential(*seq) if len(seq) > 1 else seq[0]

	def forward(self, x):
		return self.layer(x)

@fig.Component('conv-layer')
class ConvLayer(fm.Model):

	def __init__(self, A):
		# in_shape=None, out_shape=None, channels=None,
		#
		# factor=1,
		#
		# kernel_size=3, stride=None, padding=None, dilation=None,
		#
		# resize=None, norm=None, nonlin='elu', #dropout=None,
		#
		# residual=False, force_res=False, **conv_kwargs

		din = A.pull('in_shape', '<>din', None)
		channels = A.pull('channels', None)
		dout = None
		# if channels is not None:
		# 	size = A.pull('out_size', '<>size', None)
		# 	if size is not None:
		# 		dout = (channels, *size)
		dout = A.pull('out_shape', '<>dout', dout)

		assert din is not None or dout is not None, 'no input info'

		down = A.pull('down', None)
		if isinstance(down, int):
			down = down, down
		assert down is None or (down[0] >= 1 and down[1] >= 1), f'{down}'
		pool = None if down is None or (down[0] == 1 and down[1] == 1) else util.get_pooling(A.pull('pool', None), down)
		up = A.pull('up', None) if down is None else None
		if isinstance(up, int):
			up = up, up
		assert up is None or (up[0] >= 1 and up[1] >= 1), f'{up}'
		unpool = None
		size = None
		if down is None and up is None:
			size = A.pull('size', None)
			if size is not None:
				unpool = util.get_upsample(A.pull('unpool', None), size=size)
			else:
				down = (1,1)
		elif up is not None:
			unpool = util.get_upsample(A.pull('unpool', None), up=up)

		is_deconv = down is None and size is None and unpool is None

		kernel_size = A.pull('kernel_size', '<>kernel', (4,4) if is_deconv else (3,3))
		if isinstance(kernel_size, int):
			kernel_size = kernel_size, kernel_size
		padding = A.pull('padding', (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
		if isinstance(padding, int):
			padding = padding, padding
		dilation = A.pull('dilation', (1,1))
		if isinstance(dilation, int):
			dilation = dilation, dilation
		output_padding = A.pull('output_padding', (0,0)) if is_deconv else None
		if isinstance(output_padding, int):
			output_padding = output_padding, output_padding

		if is_deconv:
			stride = up
		elif down is not None and pool is None:
			stride = down
		else:
			stride = (1,1)

		stride = A.pull('stride', stride)
		if isinstance(stride, int):
			stride = stride, stride

		if din is None:
			Cout, Hout, Wout = dout
			if channels is None:
				channels = Cout

			Cin = channels

			H, W = Hout, Wout

			if down is not None and pool is not None:
				H, W = H*down[0], W*down[1]

			if is_deconv:
				if output_padding is not None:
					H, W = H - output_padding[0], W - output_padding[1]
				H, W = util.conv_size_change(H, W, kernel_size, padding, stride, dilation)
			else:
				H, W = util.deconv_size_change(H, W, kernel_size, padding, stride, dilation)

			if up is not None and unpool is not None:
				H, W = H//up[0], W//up[1]

			if size is not None:
				assert size == (H, W), f'{size} vs {(H, W)}'
				H, W = None, None
				raise NotImplementedError

			Hin, Win = H, W

		if dout is None:
			Cin, Hin, Win = din
			if channels is None:
				channels = Cin

			H, W = Hin, Win

			if size is not None:
				H, W = size

			if up is not None and unpool is not None:
				H, W = H*up[0], W*up[1]

			Cout = channels

			if is_deconv:
				H, W = util.deconv_size_change(H, W, kernel_size, padding, stride, dilation,
				                                     output_padding)
			else:
				H, W = util.conv_size_change(H, W, kernel_size, padding, stride, dilation)

			if down is not None and pool is not None:
				H, W = H//down[0], W//down[1]

			Hout, Wout = H, W

		A.push('channels', channels, silent=True)
		norm = util.get_normalization(A.pull('norm', None), channels)

		nonlin = util.get_nonlinearity(A.pull('nonlin', 'elu'))

		conv_kwargs = A.pull('conv_kwargs', {})

		residual = A.pull('residual', False)
		force_res = A.pull('force_res', False)

		print_dims = A.pull('print_dims', False)
		din, dout = (Cin, Hin, Win), (Cout, Hout, Wout)

		super().__init__(din, dout)

		self.unpool = unpool

		if is_deconv:
			self.conv = nn.ConvTranspose2d(Cin, Cout, kernel_size=kernel_size, padding=padding,
		                      stride=stride, dilation=dilation, output_padding=output_padding,
		                                   **conv_kwargs)
		else:
			self.conv = nn.Conv2d(Cin, Cout, kernel_size=kernel_size, padding=padding,
			                      stride=stride, dilation=dilation, **conv_kwargs)

		self.pool = pool
		self.norm = norm
		self.nonlin = nonlin

		self.res = force_res or (residual and Cin == Cout)
		if self.res and (Cin != Cout):
			print('WARNING: residual connections will be partial, because channels dont match')

		self.print_dims = print_dims

	def extra_repr(self):
		msg = ''
		if self.print_dims:
			msg += f'{self.din} -> {self.dout}\n'
		msg += f'residual={self.res}'
		return msg

	def forward(self, x):
		if self.unpool is not None:
			x = self.unpool(x)
		c = self.conv(x)
		if self.res:
			din, dout = x.size(1), c.size(1)
			if din > dout:
				x = c + x.narrow(1, 0, dout)
			else:
				if din < dout:
					B, _, H, W = x.size()
					x = torch.cat([x, torch.zeros(B, dout - din, H, W, device=x.device)], dim=1)

				x = c + x
		else:
			x = c

		if self.pool is not None:
			x = self.pool(x)
		if self.norm is not None:
			x = self.norm(x)
		if self.nonlin is not None:
			x = self.nonlin(x)
		return x


@fig.AutoComponent('layer-norm')
class LayerNorm(fm.Model):
	def __init__(self, din, eps=1e-5, elementwise_affine=True):
		super().__init__(din, din)
		self.norm = nn.LayerNorm(din, eps=eps,
		        elementwise_affine=elementwise_affine)

	def forward(self, x):
		return self.norm(x)

@fig.AutoComponent('interpolate')
class Interpolate(fm.Model):
	def __init__(self, din=None, size=None, scale_factor=None, mode='nearest',
	             align_corners=None, recompute_scale_factor=None):
		assert size is not None or (din is not None and scale_factor is not None)

		Cin = None
		if din is not None:
			Cin, Hin, Win = din

			if size is None:
				size = int(scale_factor*Hin), int(scale_factor*Win)

		super().__init__(din, (Cin, *size))

		self.size = size
		self.scale_factor = scale_factor
		self.mode = mode
		self.align_corners = align_corners
		self.recompute_scale_factor = recompute_scale_factor

	def forward(self, x):
		return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
		                     mode=self.mode, align_corners=self.align_corners,
		                     recompute_scale_factor=self.recompute_scale_factor)


@fig.AutoComponent('conv-lstm')
class ConvLSTM(fm.Model):
	def __init__(self, input_dim, hidden_dim, kernel_size,
				 batch_first=False, auto_reset=True,
				 peephole=True):
		super().__init__(input_dim, hidden_dim)

		self.kernel_size = kernel_size
		self.padding = int((kernel_size - 1) / 2)

		self.Wxi = nn.Conv2d(self.din, self.dout, self.kernel_size, 1, self.padding, bias=True)
		self.Whi = nn.Conv2d(self.dout, self.dout, self.kernel_size, 1, self.padding, bias=False)
		self.Wxf = nn.Conv2d(self.din, self.dout, self.kernel_size, 1, self.padding, bias=True)
		self.Whf = nn.Conv2d(self.dout, self.dout, self.kernel_size, 1, self.padding, bias=False)
		self.Wxc = nn.Conv2d(self.din, self.dout, self.kernel_size, 1, self.padding, bias=True)
		self.Whc = nn.Conv2d(self.dout, self.dout, self.kernel_size, 1, self.padding, bias=False)
		self.Wxo = nn.Conv2d(self.din, self.dout, self.kernel_size, 1, self.padding, bias=True)
		self.Who = nn.Conv2d(self.dout, self.dout, self.kernel_size, 1, self.padding, bias=False)

		if peephole: # pixelwise
			param = util.create_param(1, self.dout, 1, 1)
			param.data.zero_()
			self.register_parameter('Wci', param)
			param = util.create_param(1, self.dout, 1, 1)
			param.data.zero_()
			self.register_parameter('Wcf', param)
			param = util.create_param(1, self.dout, 1, 1)
			param.data.zero_()
			self.register_parameter('Wco', param)
		else:
			self.Wci = 0
			self.Wcf = 0
			self.Wco = 0

		self.register_buffer('hidden', torch.zeros(1, self.dout, 1, 1))
		self.register_buffer('cell', torch.zeros(1, self.dout, 1, 1))

		self.auto_reset = auto_reset
		self.batch_first = int(batch_first)

	def reset(self):
		self.hidden = torch.zeros(*((1,) + self.hidden.shape[1:]), dtype=self.hidden.dtype, device=self.hidden.device)
		self.cell = torch.zeros(*((1,) + self.cell.shape[1:]), dtype=self.cell.dtype, device=self.cell.device)

	def seq(self, xs, h=None, c=None, ret_cell=False):

		if self.batch_first:
			xs = xs.transpose(0,1)

		outs = []
		hiddens = []

		for x in xs:
			h, c = self(x,h,c, ret_cell=True)
			outs.append(h)
			hiddens.append(c)

		if self.auto_reset:
			self.reset()

		outs = torch.stack(outs, self.batch_first)
		if ret_cell:
			cells = torch.stack(outs, self.batch_first)
			return outs, cells
		return outs

	def forward(self, x, h=None, c=None, ret_cell=False):

		B, C, H, W = x.shape

		if h is None:
			if (H,W) != self.hidden.shape[-2:]:
				self.hidden = torch.zeros(1, self.dout, H, W, dtype=x.dtype, device=x.device)
			h = self.hidden

		if c is None:
			if (H,W) != self.cell.shape[-2:]:
				self.cell = torch.zeros(1, self.dout, H, W, dtype=x.dtype, device=x.device)
			c = self.cell

		ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
		cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
		cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
		co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
		ch = co * torch.tanh(cc)

		self.hidden = ch
		self.cell = cc

		if ret_cell:
			return ch, cc
		return ch

# endregion
#################
