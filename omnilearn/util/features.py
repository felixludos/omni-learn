import random
import torch

from wrapt import ObjectProxy
from omnibelt import Value as ValueWrapper, primitives, InitWall, unspecified_argument


from omnifig import Configurable

from .math import set_seed

# region Attributes

class Statelike:
	def state_dict(self, *args, **kwargs):
		try:
			return super().state_dict(*args, **kwargs)
		except AttributeError:
			raise NotImplementedError
	
	def load_state_dict(self, data, **kwargs):
		try:
			return super().load_state_dict(data, **kwargs)
		except AttributeError:
			raise NotImplementedError


class StatelikeList(Statelike, list):
	def state_dict(self):
		return [(x.state_dict() if isinstance(x, Statelike) else None) for x in self]
	
	def load_state_dict(self, data):
		for x, y in zip(self, data):
			if x is not None:
				x.load_state_dict(y)


class StatelikeDict(Statelike, dict):
	def state_dict(self):
		return {k: (v.state_dict() if isinstance(v, Statelike) else None) for k, v in self.items()}
	
	def load_state_dict(self, data):
		for k, v in data.items():
			if v is not None:
				self[k].load_state_dict(v)


# endregion

# region Values

class ValueBase(Statelike, ValueWrapper):
	
	# def __init__(self):
	
	def state_dict(self):
		return self.get()
	
	def load_state_dict(self, data):
		self.set(data)
	
	def item(self):  # compat for hyperparams
		return self.get()
	
	def __getstate__(self):
		return self.state_dict()
	
	def __setstate__(self, state):
		return self.load_state_dict(state)
	
	def __reduce__(self, *args, **kwargs):
		return type(self), (self.get(),)


class Value(ValueBase):
	def __init__(self, A, **kwargs):
		val = A.pull('value', '<>initial')
		assert val is None or isinstance(val, primitives), f'{val} should be a primitive'
		super().__init__(val)
		# self.__wrapped__ = val
		self._self_as_int = A.pull('as-int', isinstance(val, int))
		self.set(val)
		super(ObjectProxy, self).__init__(A, **kwargs)
	
	def set(self, val):
		if hasattr(self, '_self_as_int') and self._self_as_int:
			val = int(val)
		return super().set(val)


# endregion

# region Simple Features

class DimensionBase:
	din, dout = None, None
	
	def __init__(self, *args, din=None, dout=None, **kwargs):
		super().__init__(*args, **kwargs)
		if din is None:
			din = self.din
		if dout is None:
			dout = self.dout
		self.din, self.dout = din, dout
	
	def get_dims(self):
		return self.din, self.dout


class DeviceBase:
	def __init__(self, *args, device=None, **kwargs):
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		super().__init__(*args, **kwargs)
		self.device = device
	
	def get_device(self):
		return self.device
	
	def to(self, device):
		self.device = device
		try:
			super().to(device)
		except AttributeError:
			pass
	
	def cuda(self, device=None):
		self.to('cuda' if device is None else device)
	
	def cpu(self):
		self.to('cpu')




class Named(Configurable):
	def __init__(self, A, **kwargs):
		ident = A.pull('_ident', '<>__origin_key', None)
		super().__init__(A, **kwargs)
		self.name = ident
		
	def get_name(self):
		return self.name


class Priority(Configurable):
	def __init__(self, A, **kwargs):
		priority = A.pull('priority', None)
		super().__init__(A, **kwargs)
		self.priority = priority
	
	def get_priority(self):
		return self.priority


class Switchable(Configurable):
	def __init__(self, A, mode=None, **kwargs):
		if mode is None:
			mode = A.pull('mode', 'train')
		super().__init__(A, **kwargs)
		self.mode = mode
	
	def switch_to(self, mode):
		self.mode = mode
	
	def get_mode(self):
		return self.mode


class Deviced(Configurable, DeviceBase):
	def __init__(self, A, device=None, **kwargs):
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		device = A.pull('device', device)
		super().__init__(A, device=device, **kwargs)
		# self.device = device


class Checkpointable(Configurable):
	def checkpoint(self, path, ident=None):
		pass
	
	def load_checkpoint(self, path, ident=None):
		pass


class Dimensions(Configurable, DimensionBase):
	
	def __init__(self, A, din=None, dout=None, **kwargs):
		if din is None:
			din = A.pull('din', self.din)
		if dout is None:
			dout = A.pull('dout', self.dout)
		super().__init__(A, din=din, dout=dout, **kwargs)
		# self.din, self.dout = din, dout


class Seed(Configurable):
	def __init__(self, A, seed=unspecified_argument, **kwargs):
		if seed is unspecified_argument:
			seed = A.pull('seed', random.getrandbits(32))
		if seed is not None:
			set_seed(seed)
		super().__init__(A, **kwargs)
		self.seed = seed


class Seeded(Deviced):
	def __init__(self, A, seed=unspecified_argument, **kwargs):
		if seed is unspecified_argument:
			seed = A.pull('gen-seed', '<>seed', random.getrandbits(64))
		if seed is None:
			seed = random.getrandbits(64)
			
		super().__init__(A, **kwargs)
		
		# self.gen = torch.Generator(self.device)
		self.gen = torch.Generator()
		if seed is not None:
			self.seed = seed
			self.gen.manual_seed(seed)

	# def to(self, device):
	# 	gen = torch.Generator(device)
	# 	gen.set_state(self.gen.get_state())
	# 	self.gen = gen
	#
	# 	return super().to(device)

# endregion

