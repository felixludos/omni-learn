
import sys, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp
from .. import util

from .. import framework as fm
from .. import util
from . import layers as layerslib


def make_MLP(input_dim, output_dim, hidden_dims=[],
             nonlin='prelu', output_nonlin=None):
	'''
	:param input_dim: int
	:param output_dim: int
	:param hidden_dims: ordered list of int - each element corresponds to a FC layer with that width (empty means network is not deep)
	:param nonlin: str - choose from options found in get_nonlinearity(), applied after each intermediate layer
	:param output_nonlin: str - nonlinearity to be applied after the last (output) layer
	:return: an nn.Sequential instance with the corresponding layers
	'''

	flatten = False
	reshape = None

	din = input_dim
	dout = output_dim

	if isinstance(input_dim, (tuple, list)):
		flatten = True
		input_dim = int(np.product(input_dim))
	if isinstance(output_dim, (tuple, list)):
		reshape = output_dim
		output_dim = int(np.product(output_dim))

	nonlins = [nonlin] * len(hidden_dims) + [output_nonlin]
	hidden_dims = input_dim, *hidden_dims, output_dim

	layers = []
	if flatten:
		layers.append(nn.Flatten())
	for in_dim, out_dim, nonlin in zip(hidden_dims, hidden_dims[1:], nonlins):
		layers.append(nn.Linear(in_dim, out_dim))
		if nonlin is not None:
			layers.append(util.get_nonlinearity(nonlin))
	if reshape is not None:
		layers.append(layerslib.Reshaper(reshape))

	net = nn.Sequential(*layers)

	net.din, net.dout = din, dout
	return net

# WARNING: Deprecated
def make_conv_net(in_shape, channels=[], output_nonlin=None, nonlin='prelu', batch_norm=[], output_batch_norm=True,
                  kernels=[], pooling=[],
                  padding=[], strides=[], ret_output_shape=True):  # TODO: update to use plan and build
	assert len(channels) > 0
	assert len(kernels) == len(channels)
	try:
		assert len(pooling) == 0 or len(pooling) == len(channels)
	except TypeError:
		pooling = [pooling] * len(channels)
	assert len(padding) == 0 or len(padding) == len(channels)
	assert len(strides) == 0 or len(strides) == len(channels)
	try:
		assert len(batch_norm) == 0 or len(batch_norm) == len(channels)
	except TypeError:
		batch_norm = [batch_norm] * len(channels)
		batch_norm[-1] = output_batch_norm

	if len(pooling) != len(channels):
		pooling = [False] * len(channels)
	if len(padding) != len(channels):
		padding = [(k - 1) // 2 for k in kernels]
	if len(strides) != len(channels):
		strides = [1] * len(channels)
	if len(batch_norm) != len(channels):
		batch_norm = [True] * len(channels)
		batch_norm[-1] = output_batch_norm
	nonlins = [nonlin] * len(channels)
	nonlins[-1] = output_nonlin

	layers = []

	channels = [in_shape[0]] + channels
	shape = in_shape

	for in_c, out_c, pool, kernel, pad, stride, nonlin, bn in zip(channels, channels[1:], pooling, kernels, padding,
	                                                              strides, nonlins, batch_norm):
		layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad))

		C, H, W = shape
		C, H, W = out_c, (H + 2 * pad - (kernel - 1) - 1) // stride + 1, (W + 2 * pad - (kernel - 1) - 1) // stride + 1

		if pool and pool != 1:
			layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool, ceil_mode=True))
			H, W = (H + pool - 1) // pool, (W + pool - 1) // pool

		if bn:
			layers.append(nn.BatchNorm2d(out_c))
			# layers.append(nn.InstanceNorm2d(out_c))
		if nonlin is not None:
			layers.append(util.get_nonlinearity(nonlin))

		# print(H, W)
		shape = C, H, W

	net = nn.Sequential(*layers)

	if ret_output_shape:
		return net, shape

	return net

# WARNING: Deprecated
def make_deconv_net(out_shape, channels=[], upsampling='deconv', output_nonlin=None, output_batch_norm=True,
                    nonlin='prelu', batch_norm=[], kernels=[],  # TODO: update to use plan and build
                    padding=[], ups=[], ret_input_shape=True):
	assert len(channels) > 0
	assert upsampling in {'deconv', 'bilinear', 'nearest'}  # nearest probably doesn't work
	assert len(kernels) == len(channels)
	assert len(padding) == 0 or len(padding) == len(channels)
	try:
		assert len(batch_norm) == 0 or len(batch_norm) == len(channels)
	except TypeError:
		batch_norm = [batch_norm] * len(channels)
		batch_norm[-1] = output_batch_norm
	try:
		assert len(ups) == 0 or len(ups) == len(channels)
	except TypeError:
		ups = [ups] * len(channels)

	strides = ups if upsampling == 'deconv' else [1] * len(channels)

	if len(strides) != len(channels):
		strides = [2] * len(channels)
	if len(padding) != len(channels):
		padding = [(k - 1) // 2 for k in kernels]
	if len(batch_norm) != len(channels):
		batch_norm = [True] * len(channels)
		batch_norm[-1] = output_batch_norm
	if len(ups) != len(channels):
		ups = [2] * len(channels)
	nonlins = [nonlin] * len(channels)
	nonlins[-1] = output_nonlin

	layers = []

	channels = channels + [out_shape[0]]
	shape = out_shape

	for in_c, out_c, kernel, pad, stride, nonlin, bn, up in reversed(
			list(zip(channels, channels[1:], kernels, padding, strides, nonlins, batch_norm, ups))):
		if nonlin is not None:
			layers.append(util.get_nonlinearity(nonlin))

		if bn:
			layers.append(nn.BatchNorm2d(out_c))
			# layers.append(nn.InstanceNorm2d(out_c))

		C, H, W = shape

		pH, pW = H, W  # prev

		C, H, W = in_c, (H + 2 * pad - (kernel - 1) - 1) // stride + 1, \
		          (W + 2 * pad - (kernel - 1) - 1) // stride + 1

		dH, dW = (H - 1) * stride - 2 * pad + kernel, (W - 1) * stride - 2 * pad + kernel  # deconved

		opad = pH - dH, pW - dW

		if upsampling == 'deconv':
			layers.append(
				nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, output_padding=opad))

		else:
			layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad))

			layers.append(nn.Upsample(size=(pH, pW), mode=upsampling))
			C, H, W = C, (H + up - 1) // up, (W + up - 1) // up

		shape = C, H, W

	net = nn.Sequential(*layers[::-1])

	if ret_input_shape:
		return net, shape

	return net

class MLP(fm.Model):
	def __init__(self, input_dim, output_dim, **args):
		super().__init__(input_dim, output_dim)

		self.net = make_MLP(input_dim, output_dim, **args)

	def forward(self, x):
		return self.net(x)



def plan_conv(in_shape, channels, kernels=3, factors=1, strides=1, padding=None, dilations=1):
	assert len(channels) > 0
	L = len(channels)

	try:
		assert len(kernels) == L
	except TypeError:
		kernels = [kernels] * L

	try:
		assert len(factors) == L
	except TypeError:
		factors = [factors] * L

	try:
		assert len(dilations) == L
	except TypeError:
		dilations = [dilations] * L

	try:
		assert len(strides) == L
	except TypeError:
		strides = [strides] * L

	try:
		assert len(padding) == L
	except TypeError:
		padding = [padding] * L

	kernels = [((k, k) if isinstance(k, int) else k) for k in kernels]
	dilations = [((d, d) if isinstance(d, int) else d) for d in dilations]

	if padding is None:
		padding = [(k+1)//2 for k in kernels]
	padding = [((p, p) if isinstance(p, int) else p) for p in padding]
	strides = [((s, s) if isinstance(s, int) else s) for s in strides]
	factors = [((f, f) if isinstance(f, int) else f) for f in factors]

	padding = [((k[0] - 1) // 2, (k[1] - 1) // 2) if p is None else p for k, p in zip(kernels, padding)]

	channels = [in_shape[0]] + channels

	C, H, W = in_shape

	settings = []
	shapes = [in_shape]
	for ic, oc, k, p, s, f, d in zip(
			channels, channels[1:], kernels, padding, strides, factors, dilations,
	):
		settings.append({
			'in_channels': ic,
			'out_channels': oc,
			'kernel_size': k,  # should be tuple
			'stride': s,
			'padding': p,
			'dilation': d,
		})

		C = oc
		H = int((H + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
		W = int((W + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)

		# pooling
		H = (H + f[0] - 1) // f[0]
		W = (W + f[1] - 1) // f[1]

		shapes.append((C, H, W))

	return shapes, settings


def build_conv_layers(settings, factors=1, pool_type='max',
					  norm_type='batch', out_norm_type=None,
					  nonlin='elu', out_nonlin=None, residual=False):
	assert not residual

	L = len(settings)  # number of layers

	try:
		assert len(factors) == L
	except TypeError:
		factors = [factors] * L
	factors = [((f, f) if isinstance(f, int) else f) for f in factors]

	bns = [norm_type] * (L - 1) + [out_norm_type]
	nonlins = [nonlin] * (L - 1) + [out_nonlin]

	layers = []
	for params, f, bn, n in zip(settings, factors, bns, nonlins):
		layers.append(layerslib.ConvLayer(norm_type=bn, nonlin=n, pool=f,
								pool_type=pool_type, residual=residual,
								**params))

	return layers  # can be used as sequential or module list


def plan_deconv(out_shape, channels, kernels=2, factors=1, strides=1, padding=None):
	assert len(channels) > 0
	L = len(channels)

	channels = channels.copy()

	try:
		assert len(kernels) == L
	except TypeError:
		kernels = [kernels] * L

	try:
		assert len(factors) == L
	except TypeError:
		factors = [factors] * L

	try:
		assert len(strides) == L
	except TypeError:
		strides = [strides] * L

	try:
		assert len(padding) == L
	except TypeError:
		padding = [padding] * L

	kernels = [((k, k) if isinstance(k, int) else k) for k in kernels]
	padding = [((p, p) if isinstance(p, int) else p) for p in padding]
	strides = [((s, s) if isinstance(s, int) else s) for s in strides]
	factors = [((f, f) if isinstance(f, int) else f) for f in factors]

	padding = [((k[0] - 1) // 2, (k[1] - 1) // 2) if p is None else p for k, p in zip(kernels, padding)]

	channels.append(out_shape[0])

	C, H, W = out_shape

	settings = []
	shapes = [out_shape]
	for ic, oc, k, p, s, f, in reversed(list(zip(
			channels, channels[1:], kernels, padding, strides, factors,
	))):
		pH, pW = H, W  # prev

		C = ic
		H = int((H + 2 * p[0] - (k[0] - 1) - 1) / s[0] + 1)
		W = int((W + 2 * p[1] - (k[1] - 1) - 1) / s[1] + 1)

		dH = (H - 1) * s[0] - 2 * p[0] + k[0]
		dW = (W - 1) * s[0] - 2 * p[1] + k[1]

		opad = pH - dH, pW - dW

		settings.append({
			'in_channels': ic,
			'out_channels': oc,
			'kernel_size': k,  # should be tuple
			'stride': s,
			'padding': p,
			'output_padding': opad,
		})

		# pooling
		H = (H + f[0] - 1) // f[0]
		W = (W + f[1] - 1) // f[1]

		shapes.append((C, H, W))

	return shapes[::-1], settings[::-1]


def build_deconv_layers(settings, sizes=None, up_type='deconv',
						norm_type='batch', out_norm_type=None,
						nonlin='elu', out_nonlin=None, residual=False):
	assert not residual

	L = len(settings)  # number of layers

	try:
		assert len(sizes) == L
	except TypeError:
		sizes = [sizes] * L

	bns = [norm_type] * (L - 1) + [out_norm_type]
	nonlins = [nonlin] * (L - 1) + [out_nonlin]

	layers = []
	for params, sz, bn, n in zip(settings, sizes, bns, nonlins):
		layers.append(layerslib.DeconvLayer(norm_type=bn, nonlin=n, up_type=up_type, upsize=sz[-2:],
								  **params))

	return layers  # can be used as sequential or module list
