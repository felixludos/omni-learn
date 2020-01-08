

import os
import inspect
import torch
from torch import nn

from .. import util
# from ..models import unsup
from .. import framework as fm
from .. import models


_model_registry = {}
_reserved_names = {'list'}
def register_model(name, create_fn):
	assert name not in _reserved_names, '{} is reserved'.format(name)
	_model_registry[name] = create_fn

_mod_registry = {}
def register_modifier(name, mod_fn):
	assert name not in _reserved_names, '{} is reserved'.format(name)
	_mod_registry[name] = mod_fn

def create_component(info):
	assert info._type in _model_registry, 'Unknown model type (have you registered it?): {}'.format(info._type)

	create_fn = _model_registry[info._type]

	if '_mod' in info: # TODO: allow list of mods
		mod_fn = _mod_registry[info._mod]
		create_fn = mod_fn(create_fn)

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

# def _component_list(info):
# 	return [create_component(x) for x in info.components]
#
# register_model('list', _component_list)

register_model('double-enc', models.Double_Encoder)
register_model('double-dec', models.Double_Decoder)

register_modifier('normal', models.Normal_Distribized)


def _create_mlp(info): # mostly for selecting/formatting args (and creating sub components!)

	kwargs = {
		'input_dim': info.pull('input_dim', '<>din'),
		'output_dim': info.pull('output_dim', '<>dout'),
		'hidden_dims': info.pull('hidden_dims', '<>hidden_fc', []),
		'nonlin': info.pull('nonlin', 'prelu'),
		'output_nonlin': info.pull('output_nonlin', None),
	}

	model = models.make_MLP(**kwargs)

	return model
register_model('mlp', _create_mlp) # Outdated

register_model('nn', models.MLP)


class Stage_Model(fm.Schedulable, fm.Trainable_Model):
	def __init__(self, A):
		stages = A.pull('stages')

		din = A.pull('din')
		dout = A.pull('dout')

		criterion_info = A.pull('criterion', None)

		super().__init__(din, dout)

		self.stages = nn.ModuleList(self._process_stages(stages))

		self.criterion = None
		if criterion_info is not None:
			self.criterion = util.get_loss_type(criterion_info) \
				if isinstance(criterion_info, str) else util.get_loss_type(**criterion_info)  # TODO: automate

		self.set_optim(A)
		self.set_scheduler(A)


	def _create_stage(self, stage):
		return create_component(stage)
	def _process_stages(self, stages):

		N = len(stages)

		assert N > 0, 'no stages provided'

		din = self.din

		sub = []
		for i, stage in enumerate(stages):

			stage.din = din
			stage = self._create_stage(stage)

			sub.append(stage)
			din = stage.dout

		assert din == self.dout, 'dins and douts not set correctly: {} vs {}'.format(din, self.dout)

		return sub

	def forward(self, x):
		q = x
		for stage in self.stages:
			q = stage(q)
		return q


	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		# compute loss

		x = batch[0]
		if len(batch) == 2:
			y = batch[1]
		else:
			assert len(batch) == 1, 'Cant figure out how the batch is setup (you will have to implement your own _step)'
			y = x

		out.x, out.y = x, y

		pred = self(x)
		out.pred = pred

		loss = self.criterion(pred, y)
		out.loss = loss

		if isinstance(self, fm.Regularizable):
			reg = self.regularize(out)
			loss += reg

		out.loss = loss

		if self.train_me():
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out

register_model('stage', Stage_Model)

# class Reverse_Stage_Model(Stage_Model):
#
# 	def __init__(self, A):
#
# 		din = A.pull('latent_dim', '<>din', None)
# 		A.din = din
#
# 		super().__init__(A)
#
# 	def _process_stages(self, stages):
#
# 		N = len(stages)
#
# 		assert N > 0, 'no stages provided'
#
# 		din = self.din
#
# 		sub = []
# 		for i, stage in reversed(list(enumerate(stages))):
# 			stage.din = din
# 			stage = self._create_stage(stage)
#
# 			sub.append(stage)
# 			din = stage.dout
#
# 		assert din == self.dout, 'dins and douts not set correctly: {} vs {}'.format(din, self.dout)
#
# 		return sub
#
# 	def forward(self, x):
# 		q = x
# 		for stage in self.stages:
# 			q = stage(q)
# 		return q


class Trainable_Conv(fm.Schedulable, models.Conv_Encoder):
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

	if 'optim_type' in info:
		conv.set_optim(info)

	if 'scheduler_type' in info:
		conv.set_scheduler(info)

	return conv
register_model('conv', _create_conv)

class Trainable_Normal_Enc(fm.Schedulable, models.Normal_Conv_Encoder):
	pass

def _create_normal_conv(info):

	kwargs = _get_conv_args(info)

	assert kwargs['latent_dim'] is not None, 'must provide a latent_dim'

	kwargs.update({
		'min_log_std': info.pull('min_log_std', None),
	})

	conv = Trainable_Normal_Enc(**kwargs)

	if 'optim_type' in info:
		conv.set_optim(info)

	if 'scheduler_type' in info:
		conv.set_scheduler(info)

	return conv
register_model('normal-conv', _create_normal_conv)


class Trainable_Deconv(fm.Schedulable, models.Conv_Decoder):
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
	strides = info.pull('strides', 1)

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

	try:
		len(strides)
	except TypeError:
		strides = [strides]*num
	kwargs['strides'] = strides


	deconv = Trainable_Deconv(**kwargs)

	if 'optim_type' in info:
		deconv.set_optim(info)

	if 'scheduler_type' in info:
		deconv.set_scheduler(info)

	return deconv

register_model('deconv', _create_deconv)



















