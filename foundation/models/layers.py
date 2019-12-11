
import sys, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import framework as fm
from .. import util
from . import atom

#################
# Functions
#################

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

class Reshaper(nn.Module): # by default flattens

	def __init__(self, out_shape=(-1,)):
		super().__init__()

		self.out_shape = out_shape

	def extra_repr(self):
		return 'out={}'.format(self.out_shape)

	def forward(self, x):
		B = x.size(0)
		return x.view(B, *self.out_shape)


class Feature_Match(nn.Module):

	def __init__(self, layers, criterion='mse', weights=None,
	             out_wt=None, out_criterion=None,
	             model=None, reduction='mean'):
		super().__init__()

		self._features = []

		def hook(m, input, out):
			self._features.append(out)

		L = 0
		for layer in layers:
			L += 1
			layer.register_forward_hook(hook)
		self.num = L

		if weights is not None:
			assert L == len(weights), '{} != {}'.format(len(layers), len(weights))
			self.weights = weights
			# self.register_buffer('weights', weights)
		else:
			self.weights = None
		self.criterion = util.get_loss_type(criterion)
		assert reduction in {'mean', 'sum', 'none'}
		self.reduction = reduction

		self.out_wt = out_wt
		self.out_criterion = util.get_loss_type(out_criterion)
		if self.out_wt is not None and self.out_wt > 0:
			self.num += 1

		self._model = [model] # dont let pytorch treat this as a submodule

	def __len__(self):
		return self.num

	def clear(self):
		self._features.clear()

	def forward(self, p, q, model=None):

		if model is None:
			model = self._model[0]

		self.clear()
		po = model(p)
		pfs = self._features.copy()

		self.clear()
		qo = model(q)
		qfs = self._features.copy()

		self.clear()

		if self.weights is None:
			losses = [self.criterion(pf, qf) for pf, qf in zip(pfs, qfs)]
		else:
			losses = [w*self.criterion(pf, qf) for w, pf, qf in zip(self.weights, pfs, qfs)]

		if self.out_wt is not None and self.out_wt > 0:
			criterion = self.criterion if self.out_criterion is None else self.out_criterion
			out_loss = self.out_wt * criterion(po, qo)

			losses.append(out_loss)

		if self.reduction == 'none':
			return losses

		loss = sum(losses)

		if self.reduction == 'mean':
			return loss / len(losses)

		return loss


#################
# Layers
#################

class Recurrence(fm.Model):
	def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=1, output_nonlin=None,
				 rec_type='gru', dropout=0., batch_first=False, bidirectional=False, auto_reset=True):
		super().__init__(input_dim, output_dim)

		assert rec_type in {'gru', 'lstm', 'rnn'}

		self.rec_dim = hidden_dim
		if hidden_dim is None:
			self.rec_dim = output_dim

		rec_args = {'input_size': input_dim, 'hidden_size': self.rec_dim, 'num_layers': num_layers,
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
			self.out_layer = atom.make_MLP(self.rec_dim, output_dim, output_nonlin=output_nonlin)

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

class Fourier_Series(fm.Model): # TODO: generalize period to multiple din
	def __init__(self, input_dim=1, output_dim=1, order=100, periods=None):
		super().__init__(input_dim, output_dim)

		self.sin_coeff = nn.Linear(self.din * order, self.dout, bias=False)
		self.cos_coeff = nn.Linear(self.din * order, self.dout, bias=True)
		self.order = order

		if periods is not None:
			w = 2* np.pi / periods
			self.register_buffer('w', w.view(1, -1))
		else:
			w = nn.Parameter(torch.tensor([2 * np.pi] * input_dim).view(1, -1), requires_grad=True)
			self.register_parameter('w', w)

		freqs = torch.arange(1, order + 1).view(1, 1, -1).float()  # 1 x 1 x O
		self.register_buffer('freqs', freqs)

	def forward(self, x):

		x = x.view(-1, self.din).mul(self.w).unsqueeze(-1)  # B x D x 1
		x = x.mul(self.freqs).view(-1, self.din * self.order)

		return self.sin_coeff(torch.sin(x)) + self.cos_coeff(torch.cos(x))

	def get_period(self):
		return 2 * np.pi / self.w

class RunningNormalization(fm.Model):
	def __init__(self, dim, cmin=-5, cmax=5):
		super().__init__(dim, dim)
		self.dim = dim
		self.n = 0
		self.cmin, self.cmax = cmin, cmax

		self.register_buffer('sum_sq', torch.zeros(dim))
		self.register_buffer('sum', torch.zeros(dim))
		self.register_buffer('mu', torch.zeros(dim))
		self.register_buffer('sigma', torch.ones(dim))

	def update(self, xs):
		xs = xs.view(-1, self.dim)
		self.n += xs.shape[0]
		self.sum += xs.sum(0)
		self.mu = self.sum / self.n

		self.sum_sq += xs.pow(2).sum(0)
		self.mean_sum_sq = self.sum_sq / self.n

		if self.n > 1:
			self.sigma = (self.mean_sum_sq - self.mu**2).sqrt()

	def forward(self, x):
		if self.training:
			self.update(x)
		return ((x - self.mu) / self.sigma).clamp(self.cmin, self.cmax)



class PowerLinear(fm.Model):  # includes powers of input as features
	def __init__(self):
		raise Exception('not implemented')


class DenseLayer(fm.Model):
	def __init__(self, in_dim, out_dim, use_bias=True, nonlinearity='elu'):
		super(DenseLayer, self).__init__(in_dim, out_dim)
		self.layer = nn.Linear(in_dim, out_dim, bias=use_bias)
		self.nonlin = util.get_nonlinearity(nonlinearity) if nonlinearity is not None else None

	def forward(self, x):
		x = self.layer(x)
		if self.nonlin is not None:
			x = self.nonlin(x)
		return x


class ConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, pool=None, pool_type='max',
				 norm_type=None, nonlin='elu', residual=False, **conv_kwargs):
		super().__init__()

		self.res = residual
		self.conv = nn.Conv2d(in_channels, out_channels, **conv_kwargs)
		self.norm = util.get_normalization(norm_type, out_channels)
			# self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
			# self.bn = nn.InstanceNorm2d(out_channels, eps=1e-3)
		pool_type = nn.MaxPool2d if pool_type == 'max' else nn.AvgPool2d
		if pool is not None and (pool[0] > 1 or pool[1]>1):
			self.pool = pool_type(kernel_size=pool, stride=pool, ceil_mode=True)
		self.nonlin = util.get_nonlinearity(nonlin)

	def forward(self, x):
		c = self.conv(x)
		x = c + x if self.res else c

		if hasattr(self, 'pool'):
			x = self.pool(x)
		if self.norm is not None:
			x = self.norm(x)
		if self.nonlin is not None:
			x = self.nonlin(x)
		return x


class DeconvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, up_type='deconv', upsize=None, stride=1,
				 norm_type=None, nonlin='elu', output_padding=None, **conv_kwargs):
		super().__init__()

		self.up_type = up_type
		if self.up_type == 'deconv':
			self.deconv = nn.ConvTranspose2d(in_channels, out_channels, stride=stride,
											 output_padding=output_padding, **conv_kwargs)
		else:
			self.deconv = nn.Sequential(nn.Upsample(size=upsize, mode=up_type),
										nn.Conv2d(in_channels, out_channels, **conv_kwargs))

		if norm_type is not None:
			self.norm = util.get_normalization(norm_type, out_channels)
			# self.bn = nn.InstanceNorm2d(out_channels, eps=0.001)

			assert self.norm is not None, norm_type

		# print(norm_type)
		# print(hasattr(self, 'norm'))


		if nonlin is not None:
			self.nonlin = util.get_nonlinearity(nonlin)

	def forward(self, x, y=None):
		x = self.deconv(x)
		if y is not None:
			# assert x.size() == y.size(), '{} vs {}'.format(x.size(), y.size())
			x = x + y
		if hasattr(self, 'norm'):
			x = self.norm(x)
		if hasattr(self, 'nonlin'):
			x = self.nonlin(x)
		return x


class DoubleConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, factor=1, down_type='max',
	             norm_type=None, nonlin='elu', output_nonlin='default',

	             internal_channels=None, squeeze=False, residual=False,
	             ):
		super().__init__()

		assert factor in {1,2}, 'factor {} not supported'.format(factor)
		assert nonlin is not None, 'not deep'
		assert factor == 1 or not residual, 'residual requires same size after conv2'

		if internal_channels is None:
			internal_channels = out_channels

		self.conv = nn.Conv2d(in_channels, internal_channels, kernel_size=2, padding=1, stride=1)
		self.nonlin = util.get_nonlinearity(nonlin)

		self.squeeze = None
		if squeeze:
			self.squeeze = nn.Conv2d(internal_channels, out_channels, kernel_size=1, padding=0, stride=1)
			self.nonlin_squeeze = util.get_nonlinearity(nonlin)
			internal_channels = out_channels

		self.conv2 = nn.Conv2d(internal_channels, out_channels, kernel_size=2, padding=0, stride=1)

		self.residual = residual

		self.nonlin_down = util.get_nonlinearity(nonlin) if down_type == 'conv' else None
		self.down = util.get_pooling(down_type, factor, chn=out_channels)

		self.norm = util.get_normalization(norm_type, out_channels)
		if 'default' == output_nonlin:
			output_nonlin = nonlin
		self.out_nonlin = util.get_nonlinearity(output_nonlin)

	def forward(self, x):

		c = self.nonlin(self.conv(x))
		if self.squeeze is not None:
			c = self.nonlin_squeeze(self.squeeze(c))
		c = self.conv2(c)

		x = c+x if self.residual else c

		if self.down is not None:
			if self.nonlin_down is not None:
				x = self.nonlin_down(x)
			x = self.down(x)
		if self.norm is not None:
			x = self.norm(x)
		if self.out_nonlin is not None:
			x = self.out_nonlin(x)
		return x


class DoubleDeconvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, factor=None, up_type='deconv',
				 norm_type=None, nonlin='elu', output_nonlin='default',

	             internal_channels=None, squeeze=False, residual=True,
	             ):
		super().__init__()

		assert factor in {1, 2}, 'factor {} not supported'.format(factor)
		assert nonlin is not None, 'not deep'
		assert factor == 1 or not residual, 'residual requires same size after conv2'

		if internal_channels is None:
			internal_channels = out_channels
			
		self.up = util.get_upsample(up_type, factor, chn=in_channels)
		self.nonlin_up = util.get_nonlinearity(nonlin) if up_type == 'conv' else None

		self.conv = nn.Conv2d(in_channels, internal_channels, kernel_size=2, padding=1, stride=1)
		self.nonlin = util.get_nonlinearity(nonlin)

		self.squeeze = None
		if squeeze:
			self.squeeze = nn.Conv2d(internal_channels, out_channels, kernel_size=1, padding=0, stride=1)
			self.nonlin_squeeze = util.get_nonlinearity(nonlin)
			internal_channels = out_channels

		self.conv2 = nn.Conv2d(internal_channels, out_channels, kernel_size=2, padding=0, stride=1)

		self.residual = residual

		self.norm = util.get_normalization(norm_type, out_channels)
		if 'default' == output_nonlin:
			output_nonlin = nonlin
		self.out_nonlin = util.get_nonlinearity(output_nonlin)

	def forward(self, x, y=None):
		
		if self.up is not None:
			x = self.up(x)
			if self.nonlin_up is not None:
				x = self.nonlin_up(x)

		c = self.nonlin(self.conv(x))
		if self.squeeze is not None:
			c = self.nonlin_squeeze(self.squeeze(c))
		c = self.conv2(c)

		x = c+x if self.residual else c
		if y is not None:
			x += y

		if self.norm is not None:
			x = self.norm(x)
		if self.out_nonlin is not None:
			x = self.out_nonlin(x)
		return x



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

