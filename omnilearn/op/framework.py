import sys, os
from pathlib import Path
import numpy as np
import torch
import copy
import torch.nn as nn

from omnibelt import primitives, InitWall, Simple_Child, unspecified_argument

import omnifig as fig

from .. import util
from ..util import Configurable, Seed, Checkpointable, Switchable, TrackedAttrs, \
	Dimensions, Deviced, DeviceBase, DimensionBase
import torch.multiprocessing as mp
from itertools import chain

from .clock import AlertBase


class FunctionBase(DimensionBase, DeviceBase, InitWall, nn.Module):  # any differentiable vector function
	def __init__(self, din=None, dout=None, device=None, **unused):
		super().__init__(din=din, dout=dout, device=device, **unused)
	

class Function(Switchable, TrackedAttrs, Dimensions, Deviced, Configurable, FunctionBase):
	
	def switch_to(self, mode):
		super().switch_to(mode)
		if mode == 'train':
			self.train()
		else:
			self.eval()
	
	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
		
		super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs)
	
		if len(unexpected_keys): # because pytorch doesn't like persistent buffers that are None
			persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
			for name, val in persistent_buffers.items():
				if val is None:
					key = prefix + name
					if key in unexpected_keys:
						setattr(self, name, state_dict[key])
						unexpected_keys.remove(key)
				
	
	def get_hparams(self):
		return {}


class FunctionWrapperBase(Simple_Child, FunctionBase):
	def __init__(self, function, **kwargs):
		super().__init__(_parent=function, **kwargs)
		self.__dict__['_parent'] = self._parent
		del self._modules['_parent']
		self.function = function
	
	def forward(self, *args, **kwargs):
		return self.function(*args, **kwargs)
	
	
class FunctionWrapper(Configurable, FunctionWrapperBase):
	def __init__(self, A, function=unspecified_argument, **kwargs):
		if function is None:
			function = A.pull('function', None)
		super().__init__(A, function=function, **kwargs)


class HyperParam(Function):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._hparams = set()
	
	def register_hparam(self, name, val):
		# if not isinstance(val, util.ValueBase):
		# 	val = util.ValueBase(val)
		self._hparams.add(name)
		self.register_attr(name, val)
		
	def state_dict(self, *args, **kwargs):
		state = super().state_dict(*args, **kwargs)
		state['hparams'] = list(self._hparams)
		return state
		
	def load_state_dict(self, state_dict, strict=True):
		if strict:
			assert len(self._hparams) == len(state_dict['hparams']) \
			       == len(self._hparams.intersection(set(state_dict['hparams'])))
		return super().load_state_dict(state_dict, strict=strict)
	
	def get_hparams(self):
		hparams = {}
		for name in self._hparams:
			try:
				val = getattr(self, name, None)
			except AttributeError:
				continue
			hparams[name] = val if type(val) in primitives else val.item()
		return hparams


class Savable(Checkpointable, Function):
	
	def __init__(self, A, **kwargs):
		strict_load_state = A.pull('strict_load_state', True)
		
		super().__init__(A, **kwargs)
		
		self._strict_load_state = strict_load_state
	
	def checkpoint(self, path, ident='model'):
		data = {
			'model_str': str(self),
			'model_state': self.state_dict(),
		}
		path = Path(path)
		if path.is_dir():
			path = path / f'{ident}.pth.tar'
		torch.save(data, str(path))
		
	def load_checkpoint(self, path, ident='model', _data=None):
		if _data is None:
			path = Path(path)
			if path.is_dir():
				path = path / f'{ident}.pth.tar'
			data = torch.load(str(path), map_location=self.get_device())
		else:
			data = _data
		self.load_state_dict(data['model_state'], strict=self._strict_load_state)
		return path
	

		
class Recordable(util.StatsClient, Function):
	pass
	
class Maintained(Function, AlertBase):
	
	def maintain(self, tick, info=None):
		pass
	
	def activate(self, tick, info=None):
		return self.maintain(tick, info=info)

class Visualizable(Recordable):
	def visualize(self, info, logger): # records output directly to logger
		with torch.no_grad():
			self._visualize(info, logger)
			
	def _visualize(self, info, logger):
		pass # by default nothing is visualized


class Evaluatable(Recordable): # TODO: maybe not needed

	def evaluate(self, info, config=None, out=None):
		if config is None:
			config = info.get_config()
		return self._evaluate(info, config, out=out)

	def _evaluate(self, info, config, out=None):
		if out is None:
			out = util.TensorDict()
		return out # by default eval does nothing


@fig.AutoModifier('optim')
class Optimizable(Function):

	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		self.optim = None

	def set_optim(self, A=None):
		
		if self.optim is not None:
			return
		
		sub_optims = {}
		for name, child in self.named_children():
			if isinstance(child, Optimizable) and child.optim is not None:
				sub_optims[name] = child.optim

		if len(sub_optims):
			params = [] # param eters not already covered by sub_optims
			for name, param in self.named_parameters():
				name = name.split('.')[0]
				if name not in sub_optims:
					params.append(param)
		
			if len(sub_optims) == 1 and len(params) == 0: # everything convered by a single sub_optim
				optim = next(iter(sub_optims.values()))
			else:
				if len(params):
					if 'me' in sub_optims:
						raise Exception('invalid names: {} already in {}'.format('me', sub_optims.keys()))
					# sub_optims['me'] = util.default_create_optim(params, optim_info)
					sub_optims['me'] = A.pull('optim')
					sub_optims['me'].prep(params)
				optim = util.Complex_Optimizer(**sub_optims)
		else:
			optim = A.pull('optim')
			optim.prep(self.parameters())
			# optim = util.default_create_optim(self.parameters(), optim_info)
			
		self.optim = optim
		
		return optim

	def _optim_step(self, loss): # should only be called during training
		if self.optim is None:
			raise Exception('Optimizer not set')
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	def load_state_dict(self, state_dict, strict=True):
		super().load_state_dict(state_dict['model'], strict=strict)
		if self.optim is not None and 'optim' in state_dict:
			self.optim.load_state_dict(state_dict['optim'])

	def state_dict(self, *args, **kwargs):
		state_dict = {
			'model': super().state_dict(*args, **kwargs),
		}
		if self.optim is not None:
			state_dict['optim'] = self.optim.state_dict()
		return state_dict


class Trainable(Maintained, HyperParam, Recordable, Optimizable, Function, AlertBase):
	
	def __init__(self, A, **kwargs):
		
		optim_metric = A.pull('optim-metric', 'loss')
		
		super().__init__(A, **kwargs)
		
		self.optim_metric = optim_metric
		self.register_stats(self.optim_metric)
	
	def activate(self, tick, info=None):
		batch = info.get_batch()
		info.receive_output(batch, self.step(batch))
		return super().activate(tick, info=info)
	
	def step(self, batch, **kwargs):  # Override pre-processing mixins
		out = self._step(batch, **kwargs)
		# if self.train_me() and self.optim_metric in out:
		if self.optim_metric in out:
			self.mete(self.optim_metric, out[self.optim_metric])
		return out

	def _step(self, batch, out=None):  # Override post-processing mixins
		if out is None:
			out = util.TensorDict()
		return out

	def get_metric(self):
		if self.optim_metric is not None:
			return self.get_stat(self.optim_metric)

	def get_description(self):
		progress = ''
		if self.optim_metric is not None:
			metric = self.get_metric()
			if metric is not None and metric.count > 0:
				progress = f'{self.optim_metric}: {metric.val:.3f} ({metric.smooth:.3f})'
				
		return progress

	# NOTE: never call an optimizer outside of _step (not in mixinable functions)
	# NOTE: before any call to an optimizer check with self.train_me()
	def train_me(self):
		return self.training and self.optim is not None
	

class Model(Seed, Savable, Trainable, Evaluatable, Visualizable, Function): # top level - must be implemented to train
	pass




class Regularizable(object):
	def regularize(self, q):
		return torch.tensor(0).type_as(q)



@fig.AutoModifier('generative')
class Generative(object):
	def sample_prior(self, N=1):
		raise NotImplementedError

	def generate(self, N=1, prior=None):
		if prior is None:
			prior = self.sample_prior(N)
		return self(prior)


@fig.AutoModifier('encodable')
class Encodable(object):
	def encode(self, x): # by default this is just forward pass
		return self(x)


@fig.AutoModifier('decodable')
class Decodable(object): # by default this is just the forward pass
	def decode(self, q):
		return self(q)


@fig.AutoModifier('invertible')
class Invertible(object):
	def inverse(self, *args, **kwargs):
		raise NotImplementedError



