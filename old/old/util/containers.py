from pathlib import Path
import numpy as np
import torch
from collections import OrderedDict
from omnibelt import HierarchyPersistent, unspecified_argument
import omnifig as fig

from .features import DeviceBase, Statelike, Deviced

def to(obj, device):
	try:
		return obj.to(device)
	except AttributeError:
		if isinstance(obj, (list, tuple)):
			return TensorList([to(o, device) for o in obj])
		if isinstance(obj, dict):
			return TensorDict({k:to(v,device) for k,v in obj.items()})
		raise Exception('Unknown object {}: {}'.format(type(obj), obj))

class Movable(DeviceBase):
	def size(self, *args, **kwargs):
		raise NotImplementedError

class TensorDict(Movable, OrderedDict):
	def __init__(self, *args, _size_key=None, **kwargs):
		self.__dict__['device'] = None
		super().__init__(*args, **kwargs)
		self.__dict__['_size_key'] = _size_key

	def to(self, device):
		super().to(device)
		for k,v in self.items():
			try:
				self[k] = v.to(device)
			except AttributeError:
				pass

		return self

	def detach(self):
		for k,v in self.items():
			try:
				self[k] = v.detach()
			except AttributeError:
				pass

		return self

	def _find_size_key(self):
		for k, v in self.items():
			if isinstance(v, torch.Tensor):
				self._size_key = k
				return

		raise Exception('No torch.Tensor found')

	def split(self, num):
		# assert 0 < num < self.size(0)
		start, end = TensorDict(), TensorDict()
		for key, val in self.items():
			start[key] = val[:num]
			end[key] = val[num:]
		return start, end

	@staticmethod
	def merge(ds):
		if len(ds) == 1:
			return ds[0]
		merged = TensorDict()
		for key in ds[0].keys():
			merged[key] = [d[key] for d in ds]
			if isinstance(ds[0][key], torch.Tensor):
				merged[key] = torch.cat(merged[key])
		return merged

	def size(self, *args, **kwargs):
		if self._size_key is None:
			self._find_size_key()
		return self[self._size_key].size(*args, **kwargs)

	def __getstate__(self):
		return {k:v for k,v in self.items()}

	def __setstate__(self, state):
		self.update(state)

	def __getattr__(self, item):
		if item in self.__dict__ or item in {'__setattr__', '__setstate__'}:
			return super().__getattribute__(item)
		return self.__getitem__(item)

	def __setattr__(self, key, value):
		if key in self.__dict__:
			return super().__setattr__(key, value)
		return self.__setitem__(key, value)

	def __delattr__(self, item):
		if item in self.__dict__:
			return super().__delattr__(item)
		return self.__delitem__(item)

class TensorList(Movable, list):

	def __init__(self, *args, _size_key=None, **kwargs):
		super().__init__(*args, **kwargs)
		self._size_key = _size_key

	def to(self, device):
		super().to(device)
		for i,x in enumerate(self):
			try:
				self[i] = x.to(device)
			except AttributeError:
				pass

		return self

	@staticmethod
	def merge(ls):
		if len(ls) == 1:
			return ls[0]
		return TensorList([torch.cat(el) for el in zip(*ls)])

	def split(self, num):
		# assert 0 < num < self.size(0)
		return TensorList(x[:num] for x in self), TensorList(x[num:] for x in self)
		
	def _find_size_key(self):
		for i, x in enumerate(self):
			if isinstance(x, torch.Tensor):
				self._size_key = i
				return

		raise Exception('No torch.Tensor found')

	def size(self, *args, **kwargs):
		if self._size_key is None:
			self._find_size_key()
		return self[self._size_key].size(*args, **kwargs)

	def __getstate__(self):
		return [x for x in self]

	def __setstate__(self, state):
		for i,x in enumerate(state):
			if i == len(self):
				self.append(x)
			else:
				self[i] = x


#@fig.AutoModifier('smart-results')
class SmartResults(HierarchyPersistent):
	def _save_datafile(self, data, path, ext=unspecified_argument, overwrite=False,
	                   separate_dict=True, recursive=False, _save_fn=None):
		# if ext is unspecified_argument:
		if isinstance(data, str):
			ext = 'txt'
			_save_fn = save_txt
		elif isinstance(data, (np.ndarray, int, float)):
			ext = 'npy'
			_save_fn = lambda d, p: np.save(p, d)
		else:
			ext = 'pth.tar'
			_save_fn = torch.save

		return super()._save_datafile(data, path, ext=ext, overwrite=overwrite,
		                              separate_dict=separate_dict, recursive=recursive,
		                              _save_fn=_save_fn)


	def _load_datafile(self, path, ext=unspecified_argument, _load_fn=None, delimiter='/', **kwargs):
		if isinstance(path, str) and delimiter is not None:
			path = Path(*path.split(delimiter))
		path = self._get_datafile_path(path=path, ext=ext)

		_load_fn = torch.load
		if path.suffix == '.txt':
			_load_fn = load_txt
		elif path.suffix == '.npy':
			_load_fn = np.load
		return super()._load_datafile(path, ext=None, delimiter=delimiter, _load_fn=_load_fn, **kwargs)



class Cached(DeviceBase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.volatile = TensorDict()
	
	def to(self, device):
		super().to(device)
		self.volatile.to(self.get_device())


class TrackedAttrs(Statelike, Cached):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		self._saveable_attrs = set()
	
	def register_attr(self, name, data=None, update=True):
		self._saveable_attrs.add(name)
		if update:
			setattr(self, name, data)
	
	def state_dict(self, *args, **kwargs): # TODO: prevent overlapping parameters collection
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
				attrs[name] = val.state_dict() if isinstance(val, Statelike) else val

		out = {'attrs':attrs}
		if data is not None:
			out['parameters'] = data

		self.volatile = volatile
		return out

	def load_state_dict(self, state_dict, strict=True):
		missing_attrs = []
		for name, data in state_dict.get('attrs', {}).items():
			try:
				val = getattr(self, name)
			except AttributeError:
				if data is not None:
					missing_attrs.append(name)
				else:
					setattr(self, name, None)
			else:
				if isinstance(val, Statelike):
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
			raise
	
	def __setattr__(self, key, value):
		if isinstance(value, Statelike):
			self.register_attr(key, value, update=False)
		super().__setattr__(key, value)
	
	def __delattr__(self, item):
		if item in self._saveable_attrs:
			self._saveable_attrs.discard(item)
		return super().__delattr__(item)




