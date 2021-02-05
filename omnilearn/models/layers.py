import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from omnibelt import unspecified_argument
import omnifig as fig

# from .. import framework as fm
from ..op import framework as fm
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
class Recurrence(fm.FunctionBase):
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
class FourierLayer(fm.FunctionBase): # TODO: generalize period to multiple din
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


class Invertible(fm.FunctionBase):
	def __init__(self, features, bias=True):
		super().__init__(features, features)
		
		self.features = features
		
		self.wU = nn.Parameter(torch.Tensor(features, features))
		self.wV = nn.Parameter(torch.Tensor(features, features))
		self.log_S = nn.Parameter(torch.Tensor(features))
		
		if bias:
			self.bias = nn.Parameter(torch.Tensor(features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
	
	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.wU, a=math.sqrt(5))
		nn.init.kaiming_uniform_(self.wV, a=math.sqrt(5))
		
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wV)
		bound = 1 / math.sqrt(fan_in)
		nn.init.uniform_(self.log_S, -bound, bound)
		
		if self.bias is not None:
			nn.init.uniform_(self.bias, -bound, bound)
		
	def _svd(self, inverse=False):
		U = util.orthogonalize(self.wU)
		V = util.orthogonalize(self.wV)
		log_S = self.log_S
		
		if inverse:
			V, U = U.t(), V.t()
			log_S = -log_S
			
		return U, log_S.exp(), V
		
	def _mat(self, inverse=False):
		U, S, V = self._svd(inverse=inverse)
		return (U * S.unsqueeze(0)) @ V
		
		
	def forward(self, input):
		return F.linear(input, self._mat(), self.bias)
		
	def inverse(self, input):
		if self.bias is not None:
			input = input - self.bias.unsqueeze(0)
		return F.linear(input, self._mat(True))
	
	def extra_repr(self):
		return f'features={self.features}, bias={self.bias is not None}'

@fig.AutoComponent('inv-layer')
class InvertibleLayer(Invertible):
	def __init__(self, A):
		dim = A.pull('dim', '<>features', None)
		if dim is None:
			dim = A.pull('din', None)
		if dim is None:
			dim = A.pull('dout', None)
		if dim is None:
			raise Exception('unknown size')
		
		bias = A.pull('bias', True)
		
		super().__init__(dim, bias=bias)


class PowerLinear(fm.FunctionBase):  # includes powers of input as features
	def __init__(self):
		raise NotImplementedError


class DenseLayerBase(fm.FunctionBase):
	def __init__(self, din, dout, bias=True, nonlin='elu', **kwargs):
		super().__init__(din=din, dout=dout, **kwargs)

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


@fig.Component('dense-layer')
class DenseLayer(fm.Function, DenseLayerBase):
	def __init__(self, A, din=None, dout=None, bias=None, nonlin=None, **kwargs):
		
		if din is None:
			din = A.pull('din')
			if din is None:
				din = A.pull('width', '<>channels')
		if dout is None:
			dout = A.pull('dout')
			if dout is None:
				dout = A.pull('width', '<>channels')
		if bias is None:
			bias = A.pull('bias', True)
		if nonlin is None:
			nonlin = A.pull('nonlin', 'elu')
		
		super().__init__(A, din=din, dout=dout, bias=bias, nonlin=nonlin, **kwargs)


class ConvLayerBase(fm.FunctionBase):
	def __init__(self, cin, cout, kernel_size,
	             padding=None, stride=1, dilation=1, output_padding=0,
	             unpool=None, pool=None, norm=None, nonlin=None,
	             residual=False, force_res=False, transpose=False, print_dims=True,
	             conv_kwargs={}, **kwargs):
		
		super().__init__(**kwargs)
		
		self.unpool = unpool
		
		if transpose:
			self.conv = nn.ConvTranspose2d(cin, cout, kernel_size=kernel_size, padding=padding,
			                               stride=stride, dilation=dilation, output_padding=output_padding,
			                               **conv_kwargs)
		else:
			self.conv = nn.Conv2d(cin, cout, kernel_size=kernel_size, padding=padding,
			                      stride=stride, dilation=dilation, **conv_kwargs)
		
		self.pool = pool
		self.norm = norm
		self.nonlin = util.get_nonlinearity(nonlin)
		
		self.res = (force_res or (residual and cin == cout)) \
		           and (self.conv.stride[0] == self.conv.stride[1] == 1)
		if force_res and not self.res:
			print(f'WARNING: cant use residual due to stride {stride}')
		if self.res and (cin != cout):
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




@fig.Component('conv-layer')
class ConvLayer(fm.Function, ConvLayerBase):

	def __init__(self, A, din=None, dout=None, channels=None,
	             down=None, up=None, size=None, transpose=None,
	             **kwargs):

		if din is None:
			din = A.pull('in-shape', '<>din', None)
		if channels is None:
			channels = A.pull('channels', None)
		if dout is None:
			dout = A.pull('out-shape', '<>dout', None)
			
		if transpose is None:
			transpose = A.pull('transpose', None)

		assert din is not None or dout is not None, 'no input info'

		if din is not None and dout is not None:

			Cin, Hin, Win = din
			Cout, Hout, Wout = dout

			if channels is None:
				channels = Cout

			if Hin != Hout or Win != Wout:
				H, W = Hout/Hin, Wout/Win

				val = None
				is_down = not ((H >= 1 and W >= 1) or (H < 1 and W > 1) or (H > 1 and W < 1))
				if is_down:
					H, W = 1/H, 1/W
				if int(H) == H and int(W) == W:
					val = int(H), int(W)
				else:
					size = Hout, Wout
				down = val if is_down else None
				up = None if is_down else val
		else:
			if down is None:
				down = A.pull('down', down)
			if isinstance(down, int):
				down = down, down
			assert down is None or (down[0] >= 1 and down[1] >= 1), f'{down}'
			if up is None and transpose is not False:
				up = A.pull('up', up) #if down is None else None
			if isinstance(up, int):
				up = up, up
			if down is None and up is not None and up[0] == up[1] == 1 and transpose is None:
				down = up
				up = None
			assert up is None or (up[0] >= 1 and up[1] >= 1), f'{up}'
			if down is None and up is None:
				if size is None:
					size = A.pull('size', None)
				if size is None:
					down = (1,1)
					
			if down is not None and up is not None:
				down = down if not transpose else None
				up = up if transpose else None

		pool = None if down is None or (down[0] == down[1] == 1) \
			else util.get_pooling(A.pull('pool', None), down)
		# A.begin()
		A.push('channels', channels, silent=True)
		unpool = util.get_upsample(A.pull('unpool', None), size=size) if size is not None \
			else (util.get_upsample(A.pull('unpool', None), up=up, channels=channels)
			      if up is not None and (up[0] > 1 or up[1] > 1) else None)
		# A.abort()
		if transpose: # make sure transpose is valid
			transpose = down is None or up is not None
		is_deconv = size is None and unpool is None and (transpose or (up is not None and down is None))
		# is_deconv = down is None and size is None and unpool is None

		kernel_size = A.pull('kernel_size', '<>kernel', (3+up[0]-1, 3+up[1]-1) if is_deconv else (3,3)) # TODO: check condition for larger default kernel for deconvs
		if kernel_size is None:
			kernel_size = (4,4) if is_deconv else (3,3)
		if isinstance(kernel_size, int):
			kernel_size = kernel_size, kernel_size
		padding = A.pull('padding', (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
		if padding is None:
			padding = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
		if isinstance(padding, int):
			padding = padding, padding
		dilation = A.pull('dilation', (1,1))
		if dilation is None:
			dilation = (1,1)
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

		_stride = stride
		stride = A.pull('stride', stride)
		if stride is None:
			stride = _stride
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

		# A.begin()
		A.push('channels', Cout, silent=True)
		norm = util.get_normalization(A.pull('norm', None), Cout)
		# A.abort()

		nonlin = util.get_nonlinearity(A.pull('nonlin', 'elu'))

		conv_kwargs = A.pull('conv_kwargs', {})

		residual = A.pull('residual', False)
		force_res = A.pull('force_res', False)

		print_dims = A.pull('print_dims', False)
		din, dout = (Cin, Hin, Win), (Cout, Hout, Wout)

		super().__init__(A, din=din, dout=dout,
		                 cin=Cin, cout=Cout, kernel_size=kernel_size,
	             padding=padding, stride=stride, dilation=dilation, output_padding=output_padding,
	             unpool=unpool, pool=pool, norm=norm, nonlin=nonlin,
	             residual=residual, force_res=force_res, transpose=is_deconv, print_dims=print_dims,
	             conv_kwargs=conv_kwargs, **kwargs)


class ExpandedConvLayer(ConvLayer):
	def __init__(self, A, din=None, dout=None, channels=None, extra_channels=0, **kwargs):
		if din is None:
			din = A.pull('in-shape', '<>din', None)
		if channels is None:
			channels = A.pull('channels', None)
		if dout is None:
			dout = A.pull('out-shape', '<>dout', None)
		
		assert din is not None or dout is not None, 'no input info'
		
		if din is not None and dout is not None:
			Cin, Hin, Win = din
			Cout, Hout, Wout = dout
			if channels is None:
				channels = Cout
		
		if din is not None:
			Cin, Hin, Win = din
			Cin += extra_channels
			din = Cin, Hin, Win
		else:
			if channels is None:
				channels = dout[0]
			channels += extra_channels
		
		super().__init__(A, din=din, dout=dout, channels=channels, **kwargs)
		
	def preprocess(self, x):
		return x
		
	def forward(self, x):
		x = self.preprocess(x)
		return super().forward(x)
	
	
@fig.Component('coord-conv-layer')
class CoordConvLayer(ExpandedConvLayer):
	def __init__(self, A, din=None, dout=None, channels=None, extra_channels=0, **kwargs):
		super().__init__(A, din=din, dout=dout, channels=channels,
		                 extra_channels=extra_channels+2, **kwargs)
		
		C, H, W = self.din
		Hax = torch.linspace(0, 1, H)
		Wax = torch.linspace(0, 1, W)
		coords = torch.stack(torch.meshgrid(Hax, Wax), 0)
		self.register_buffer('coords', coords)
	
	def preprocess(self, x):
		x = super().preprocess(x)
		coords = self.coords.unsqueeze(0).expand(x.size(0), *self.coords.shape)
		return torch.cat([x, coords], 1)


@fig.AutoComponent('layer-norm')
class LayerNorm(fm.FunctionBase):
	def __init__(self, din, eps=1e-5, elementwise_affine=True):
		super().__init__(din, din)
		self.norm = nn.LayerNorm(din, eps=eps,
				elementwise_affine=elementwise_affine)

	def forward(self, x):
		return self.norm(x)


@fig.AutoComponent('interpolate')
class Interpolate(fm.FunctionBase):
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
class ConvLSTM(fm.FunctionBase):
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
