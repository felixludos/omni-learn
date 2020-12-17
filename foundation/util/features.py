
import torch

from wrapt import ObjectProxy
from omnibelt import Value as ValueWrapper, primitives

from .containers import TensorDict

# region Attributes

class Attribute:
	def state_dict(self):
		raise NotImplementedError
	
	def load_state_dict(self, data):
		raise NotImplementedError


class AttributeList(Attribute, list):
	def state_dict(self):
		return [(x.state_dict() if isinstance(x, Attribute) else None) for x in self]
	
	def load_state_dict(self, data):
		for x, y in zip(self, data):
			if x is not None:
				x.load_state_dict(y)


class AttributeDict(Attribute, dict):
	def state_dict(self):
		return {k: (v.state_dict() if isinstance(v, Attribute) else None) for k, v in self.items()}
	
	def load_state_dict(self, data):
		for k, v in data.items():
			if v is not None:
				self[k].load_state_dict(v)

# endregion

# region Values

class ValueBase(Attribute, ValueWrapper):
	
	def state_dict(self):
		return self.get()
	
	def load_state_dict(self, data):
		self.set(data)
	
	def item(self):  # compat for hyperparams
		return self.get()


class Value(ValueBase):
	def __init__(self, A, **kwargs):
		val = A.pull('value', '<>initial')
		assert val is None or isinstance(val, primitives), f'{val} should be a primitive'
		self.set(val)
		self.as_int = A.pull('as-int', isinstance(self.get(), int))
		super(ObjectProxy, self).__init__(A, **kwargs)
	
	def set(self, val):
		if self.as_int:
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

class Configurable:
	def __init__(self, A, _req_kwargs={}, **kwargs):
		only_req = A.pull('only-req', True, silent=True)
		if only_req:
			super().__init__(**_req_kwargs)
		else:
			super().__init__(**kwargs)


class Switchable(Configurable):
	def __init__(self, A, **kwargs):
		mode = A.pull('mode', 'train', overwrite=False)
		super().__init__(A, **kwargs)
		self.mode = mode
	
	def switch_mode(self, mode):
		self.mode = mode
	
	def get_mode(self):
		return self.mode


class Deviced(Configurable, DeviceBase):
	def __init__(self, A, **kwargs):
		device = A.pull('device', 'cuda' if torch.cuda.is_available() else 'cpu')
		super().__init__(A, devie=device, **kwargs)


class Checkpointable(Configurable):
	def checkpoint(self, path, ident=None):
		pass

	def load_checkpoint(self, path, ident=None):
		pass


class Dimensions(Configurable, DimensionBase):
	
	def __init__(self, A, din=None, dout=None, **kwargs):
		if din is None:
			din = self.din
		din = A.pull('din', din)
		if dout is None:
			dout = self.dout
		dout = A.pull('dout', dout)
		super().__init__(A, din=din, dout=dout, **kwargs)

class Cached(Deviced):
	def __init__(self, A, **kwargs):
		self.volatile = TensorDict()
		super().__init__(A, **kwargs)
		
	def to(self, device):
		super().to(device)
		self.volatile.to(self.get_device())
	
		
class TrackedAttrs(Cached):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		self._saveable_attrs = set()
	
	def register_attr(self, name, data=None, update=True):
		self._saveable_attrs.add(name)
		if update:
			setattr(self, name, data)
	
	def state_dict(self, *args, **kwargs):
		volatile = self.volatile
		self.volatile = None
		
		try:
			data = super().state_dict(*args, **kwargs)
		except AttributeError:
			data = None
		
		attrs = {}
		for name in self._saveable_attrs:
			try:
				val = getattr(self, name, None)
			except AttributeError:
				pass
			else:
				attrs[name] = val.state_dict() if isinstance(val, Attribute) else val
		
		self.volatile = volatile
		return {'parameters': data, 'attrs': attrs}
	
	def load_state_dict(self, state_dict, strict=True):
		missing_attrs = []
		for name, data in state_dict.get('attrs', {}).items():
			try:
				val = getattr(self, name)
			except AttributeError:
				missing_attrs.append(name)
			else:
				if isinstance(val, Attribute):
					val.load_state_dict(data)
				else:
					setattr(self, name, data)
		if strict and len(missing_attrs):
			raise RuntimeError('Missing registered attributes: {}'.format(', '.join(missing_attrs)))
		try:
			return super().load_state_dict(state_dict['parameters']
			                               if 'parameters' in state_dict
			                               else state_dict, strict=strict)
		except AttributeError:
			pass
	
	def __setattr__(self, key, value):
		if isinstance(value, Attribute):
			self.register_attr(key, value, update=False)
		super().__setattr__(key, value)
	
	def __delattr__(self, item):
		if item in self._saveable_attrs:
			self._saveable_attrs.discard(item)
		return super().__delattr__(item)


class StatsClient(Configurable):
	def __init__(self, A, **kwargs):
		stats = A.pull('stats', None, ref=True)
		if stats is None:
			print('WARNING: no stats manager found')
		fmt = A.pull('stats-fmt', None)
		super().__init__(A, **kwargs)
		
		self._stats = stats
		self._stats_fmt = fmt
	
	def register_stats(self, *names):
		if self._stats is not None:
			if self._stats_fmt is not None:
				names = [self._stats_fmt.format(name) for name in names]
			self._stats.new(*names)
	
	def update_stat(self, name, val, n=1):
		if self._stats is not None:
			if self._stats_fmt is not None:
				name = self._stats_fmt.format(name)
			self._stats.update(name, val, n=n)

	def get_stat(self, name):
		if self._stats is not None:
			if self._stats_fmt is not None:
				name = self._stats_fmt.format(name)
			return self._stats.get(name, None)
			

# endregion

