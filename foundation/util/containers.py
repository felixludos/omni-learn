
import torch
from collections import OrderedDict


def to(obj, device):
	try:
		return obj.to(device)
	except AttributeError:
		if isinstance(obj, (list, tuple)):
			return TensorList([to(o, device) for o in obj])
		if isinstance(obj, dict):
			return TensorDict({k:to(v,device) for k,v in obj.items()})
		raise Exception('Unknown object {}: {}'.format(type(obj), obj))

class Movable(object):

	def to(self, device):
		raise NotImplementedError

	def size(self, *args, **kwargs):
		raise NotImplementedError

class TensorDict(Movable, OrderedDict):

	def __init__(self, *args, _size_key=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.__dict__['_size_key'] = _size_key


	def to(self, device):
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

	def size(self, *args, **kwargs):
		if self._size_key is None:
			self._find_size_key()
		return self[self._size_key].size(*args, **kwargs)

	def __getattr__(self, item):
		if item in self.__dict__:
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
		for i,x in enumerate(self):
			try:
				self[i] = x.to(device)
			except AttributeError:
				pass

		return self

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



