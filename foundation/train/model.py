

import os
import inspect
import torch
# from ..models import unsup
from .. import framework as fm
from .. import models


_model_registry = {}
def register_model(name, create_fn):
	_model_registry[name] = create_fn

def create_component(info):
	assert info._type in _model_registry, 'Unknown model type (have you registered it?): {}'.format(info._type)

	create_fn = _model_registry[info._type]
	model = create_fn(info)
	return model

def default_create_model(A):

	assert '_type' in A.model

	print('Model-type: {}'.format(A.model._type))

	model = create_component(A.model)
	return model

class MissingConfigError(Exception):
	def __init__(self, key):
		super().__init__(key)


# register standard components


def _create_mlp(info): # mostly for selecting/formatting args (and creating sub components!)

	kwargs = {
		'input_dim': info.pull('input_dim', '<>din'),
		'output_dim': info.pull('output_dim', '<>dout'),
		'hidden_dims': info.pull('hidden_fc', []),
		'nonlin': info.pull('nonlin', 'prelu'),
		'output_nonlin': info.pull('output_nonlin', None),
	}

	model = models.make_MLP(**kwargs)

	return model
register_model('mlp', create_fn=_create_mlp)


class Trainable_Conv(fm.Optimizable, models.Conv_Encoder):
	pass

def _get_conv_args(info):
	kwargs = {

		# req
		'in_shape': info.pull('in_shape', '<>din'),

		'channels': info.pull('channels'),

		# optional
		'latent_dim': info.pull('latent_dim', '<>dout', None),
		'feature_dim': info.pull('feature_dim', None),

		'nonlin': info.pull('nonlin', 'prelu'),
		'output_nonlin': info.pull('output_nonlin', None),

		'down': info.pull('downsampling', 'max'),

		'norm_type': info.pull('norm_type', 'batch'),
		'output_norm_type': info.pull('output_norm_type', None),

		'hidden_fc': info.pull('hidden_fc', '<>fc', []),
	}

	num = len(kwargs['channels'])

	kernels = info.pull('kernels', 3)
	strides = info.pull('strides', 1)
	factors = info.pull('factors', 2)

	# TODO: deal with rectangular inputs
	try:
		len(kernels)
	except TypeError:
		kernels = [kernels] * num
	kwargs['kernels'] = kernels

	try:
		len(strides)
	except TypeError:
		strides = [strides] * num
	kwargs['strides'] = strides

	try:
		len(factors)
	except TypeError:
		factors = [factors] * num
	kwargs['factors'] = factors

	return kwargs

def _create_conv(info):

	kwargs = _get_conv_args(info)

	conv = Trainable_Conv(**kwargs)

	optim_info = info.pull('optim', None)
	if optim_info is not None:
		conv.set_optim(optim_info)

	return conv
register_model('conv', create_fn=_create_conv)

class Trainable_Normal_Enc(fm.Optimizable, models.Normal_Conv_Encoder):
	pass

def _create_normal_conv(info):

	kwargs = _get_conv_args(info)

	assert kwargs['latent_dim'] is not None, 'must provide a latent_dim'

	kwargs.update({
		'min_log_std': info.pull('min_log_std', None),
	})

	conv = Trainable_Normal_Enc(**kwargs)

	optim_info = info.pull('optim', None)
	if optim_info is not None:
		conv.set_optim(optim_info)

	return conv
register_model('normal-conv', create_fn=_create_normal_conv)


class Trainable_Deconv(fm.Optimizable, models.Conv_Decoder):
	pass

def _create_deconv(info):

	kwargs = {

		# req
		'out_shape': info.pull('out_shape', '<>dout'),

		'channels': info.pull('channels'),

		# optional
		'latent_dim': info.pull('latent_dim', '<>dout', None),

		'nonlin': info.pull('nonlin', 'prelu'),
		'output_nonlin': info.pull('output_nonlin', None),

		'upsampling': info.pull('upsampling', 'max'),

		'norm_type': info.pull('norm_type', 'instance'),
		'output_norm_type': info.pull('output_norm_type', None),

		'hidden_fc': info.pull('hidden_fc', '<>fc', []),
	}

	num = len(kwargs['channels'])

	kernels = info.pull('kernels', 3)
	factors = info.pull('factors', 2)

	# TODO: deal with rectangular inputs
	try:
		len(kernels)
	except TypeError:
		kernels = [kernels]*num
	kwargs['kernels'] = kernels

	try:
		len(factors)
	except TypeError:
		factors = [factors] * num
	kwargs['ups'] = factors


	deconv = Trainable_Deconv(**kwargs)

	optim_info = info.pull('optim', None)
	if optim_info is not None:
		deconv.set_optim(optim_info)

	return deconv

register_model('deconv', create_fn=_create_deconv)



















