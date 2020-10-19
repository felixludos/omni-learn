
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


def make_MLP(input_dim, output_dim, hidden_dims=None,
             initializer=None,
             nonlin='prelu', output_nonlin=None,
			 logify_in=False, unlogify_out=False,
             bias=True, output_bias=None):
	'''
	:param input_dim: int
	:param output_dim: int
	:param hidden_dims: ordered list of int - each element corresponds to a FC layer with that width (empty means network is not deep)
	:param nonlin: str - choose from options found in get_nonlinearity(), applied after each intermediate layer
	:param output_nonlin: str - nonlinearity to be applied after the last (output) layer
	:param logify_in: convert input to logified space
	:param unlogify_out: convert output back to linear space
	:return: an nn.Sequential instance with the corresponding layers
	'''

	if hidden_dims is None:
		hidden_dims = []

	if output_bias is None:
		output_bias = bias

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
	biases = [bias] * len(hidden_dims) + [output_bias]
	hidden_dims = input_dim, *hidden_dims, output_dim

	layers = []
	if flatten:
		layers.append(nn.Flatten())

	if logify_in:
		layers.append(util.Logifier())
	for in_dim, out_dim, nonlin, bias in zip(hidden_dims, hidden_dims[1:], nonlins, biases):
		layer = nn.Linear(in_dim, out_dim, bias=bias)
		if initializer is not None:
			layer = initializer(layer, nonlin)
		layers.append(layer)
		if nonlin is not None:
			layers.append(util.get_nonlinearity(nonlin))

	if unlogify_out:
		layers.append(util.Unlogifier())
	if reshape is not None:
		layers.append(layerslib.Reshaper(reshape))


	net = nn.Sequential(*layers)

	net.din, net.dout = din, dout
	return net



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

	channels = [in_shape[0]] + list(channels)

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
	# assert not residual

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
		layers.append(layerslib.ConvLayer(norm=bn, nonlin=n, factor=f,
								pool=pool_type, residual=residual,
								down_type='stride' if f == 1 else 'pool',
								**params))

	return layers  # can be used as sequential or module list


def plan_deconv(out_shape, channels, kernels=2, factors=1, strides=1, padding=None):
	assert len(channels) > 0
	L = len(channels)

	channels = list(channels)

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
						norm_type='batch', out_norm_type=None, factors=None,
						nonlin='elu', out_nonlin=None, residual=False):
	# assert not residual

	L = len(settings)  # number of layers

	try:
		assert len(sizes) == L
	except TypeError:
		sizes = [sizes] * L

	try:
		assert len(factors) == L
	except TypeError:
		factors = [factors] * L

	bns = [norm_type] * (L - 1) + [out_norm_type]
	nonlins = [nonlin] * (L - 1) + [out_nonlin]

	layers = []
	for params, f, sz, bn, n in zip(settings, factors, sizes, bns, nonlins):
		layers.append(layerslib.DeconvLayer(norm=bn, nonlin=n, up_type=up_type, size=sz[-2:],
		                                    residual=residual, factor=f,
								  **params))

	return layers  # can be used as sequential or module list


