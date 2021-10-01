
import sys, os
from pathlib import Path
from collections import OrderedDict
from wrapt import ObjectProxy
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset, TensorDataset
from torch.utils.data.dataloader import default_collate as pytorch_collate
from torch.utils.data._utils.collate import default_convert as pytorch_convert
import h5py as hf

from omnibelt import unspecified_argument, InitWall

# from .collate import SampleFormat
from .loaders import Featured_DataLoader

from .. import util

class DataLike:
	pass

# region DatasetBases

class MissingDataError(Exception):
	def __init__(self, *names):
		super().__init__(', '.join(names))
		self.names = names


class DatasetBase(DataLike, util.DimensionBase, InitWall, PytorchDataset):
	_available_modes = {'train'}
	_available_data = {}
	_sample_format = None

	def __init_subclass__(cls, available_data={}, available_modes=[], sample_format=None,  **kwargs):
		super().__init_subclass__(**kwargs)

		if sample_format is None:
			sample_format = []
		cls._sample_format = sample_format
		cls._available_modes = {*available_modes, *cls._available_modes}
		cls._available_data =  {**available_data, **cls._available_data}
		for key in cls._sample_format:
			if key not in cls._available_data:
				cls._available_data[key] = None


	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._available_modes = self.__class__._available_modes.copy()
		self._available_data = self.__class__._available_data.copy()
		self._sample_format = self.__class__._sample_format.copy()


	def get(self, name=None, idx=None, format=None, **kwargs):
		if name is not None:
			return getattr(self, f'get_{name}')(idx=idx, **kwargs)
		if format is None:
			format = self._sample_format
		if format is None or len(format) == 0:
			format = list(key for key in self._available_data if self._available_data[key] is None)
		keys = [format] if isinstance(format, str) else format
		return self.format({key: self.get(name=key, idx=idx) for key in keys}, format=format)


	def _default_get(self, name, idx=None, **kwargs):
		alias = self._available_data.get(name, None)
		if alias is None:
			data = getattr(self, name, None)  # check for tensor
			if data is not None:
				return data if idx is None else data[idx]
			raise MissingDataError(name)

		getter = getattr(self, f'get_{alias}', None)
		if getter is None:
			return self._default_get(alias, idx=idx, **kwargs)
		return getter(idx=idx, **kwargs)


	def to_loader(self, *args, format=None, **kwargs):
		if format is None:
			format = self._sample_format
		return Featured_DataLoader(self, *args, sample_format=format, **kwargs)


	def allow_batched_get(self):
		return False


	def __len__(self):
		for key in self._sample_format:
			data = getattr(self, key, None)
			if data is not None:
				return len(data)


	def __getitem__(self, item):
		return self.get(idx=item)


	def register_data(self, name, data=None, strict=True): # either provide the tensor, or implement get_{data}
		if data is not None:
			setattr(self, name, data)
			self._available_data[name] = None
		elif strict and not (hasattr(self, f'get_{name}') or hasattr(self, name)):
			raise MissingDataError(name)
		if name not in self._available_data:
			self._available_data[name] = None


	def register_data_aliases(self, base, *aliases):
		# if base not in self._available_data:
		# 	raise MissingDataError(base)
		for alias in aliases:
			self._available_data[alias] = base


	def get_available_data(self):
		return set(self._available_data.keys())


	def _filter_sample_format(self, names):
		new, missing = [], []
		for name in names:
			(new if name in self._available_data else missing).append(name)
		if len(missing):
			raise MissingDataError(*missing)
		return new


	def set_sample_format(self, *names):
		names = self._filter_sample_format(names)
		self._sample_format.clear()
		self._sample_format.extend(names)
		return self._sample_format


	def include_sample_data(self, *names):
		for name in names:
			if name not in self._available_data:
				self.register_data(name, strict=False)
		self._sample_format.extend(name for name in self._filter_sample_format(names))
		return self._sample_format


	def collate(self, samples):
		return pytorch_collate(samples)


	def format(self, sample, format=None):
		if format is None:
			format = self._sample_format

		if isinstance(format, str):
			return sample[format]
		if isinstance(format, set):
			return {key:sample[key] for key in format}
		return tuple(sample[key] for key in format)


	@classmethod
	def add_available_modes(cls, *modes):
		cls._available_modes.update(modes)


	@classmethod
	def get_available_modes(cls):
		return cls._available_modes


	# def __getattr__(self, item):
	#
	# 	try:
	# 		return super().__getattribute__(item)
	# 	except AttributeError:
	# 		if item.startswith('get_'):
	# 			name = item[4:]
	# 			if name not in self._available_data:
	# 				raise
	# 			alias = self._available_data[name]
	# 			if alias is not None:
	# 				return getattr(self, f'get_{alias}')
	# 			data = getattr(self, name, None)  # check for tensor
	# 			if data is not None:
	# 				return lambda idx, **_: data if idx is None else data[idx]
	# 			raise MissingDataError(name)
	#
	#
	# 			# alias = name
	# 			# while alias is not None and name in self._available_data:
	# 			# 	name = alias
	# 			# 	alias = self._available_data[name]
	# 			# getter = getattr(self, f'get_{name}', None)
	# 			# if getter is None:
	# 			# 	data = getattr(self, name, None)  # check for tensor
	# 			# 	if data is not None:
	# 			# 		getter = lambda idx, **_: data if idx is None else data[idx]
	# 			# if getter is not None:
	# 			# 	return getter(idx, **kwargs)
	# 			# raise MissingDataError(name)
	#
	# 			pass



class DevicedBase(util.DeviceBase, DatasetBase):
	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)
		self._buffers = []


	def register_buffer(self, name, buffer):
		'''
		By convention, all input buffers should be registered before output (label) buffers.
		This will register the buffer and set the `buffer` as an attribute of `self` with key `name`.

		:param name: name of this buffer
		:param buffer: torch.Tensor with data
		:return:
		'''
		self._buffers.append(name)
		return self.register_data(name, data=buffer)


	def to(self, device):
		super().to(device)
		buffers = getattr(self, '_buffers', None)
		if buffers is not None:
			for name in self._buffers:
				try:
					val = getattr(self, name)
					if val is not None:
						new = val.to(device)
						if new is not None:
							self.__setattr__(name, new)
				except AttributeError:
					pass



class Batchable(DevicedBase): # you can select using a full batch
	def allow_batched_get(self):
		return True


	def collate(self, samples):
		return pytorch_convert(samples)



# class Accessible(DatasetBase):
# 	def _get_all(self, key):
# 		raise NotImplementedError


class DatasetLocked(Exception):
	def __init__(self):
		super().__init__()



class Lockable(DatasetBase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._is_locked = False


	def _check_lock(self):
		if self.is_locked():
			raise DatasetLocked


	def is_locked(self):
		return self._is_locked


	def set_lock(self, l=None):
		if l is None:
			l = not self._is_locked
		self._is_locked = l
		return l


	def update_data(self, indices, **kwargs):
		self._check_lock()
		return self._update_data(indices, **kwargs)


	def _update_data(self, indices, **kwargs):
		for key in self.get_available_data():
			data = getattr(self, key, None)
			fn = getattr(self, f'_update_{key}', None)
			if data is not None:
				setattr(self, key, data[indices])
			elif fn is not None:
				fn(indices, **kwargs)



class Observation(Lockable):
	# _sample_format = ['observations']
	_observation_space = None

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.include_sample_data('observations')


	def get_observations(self, idx=None, **kwargs):
		return self._default_get('observations', idx=idx, **kwargs)


	def get_observation_space(self):
		return self._observation_space



class Supervised(Observation):
	_target_space = None
	_target_names = None

	def __init__(self, supervised=True, **kwargs):
		super().__init__(**kwargs)
		self._is_supervised = supervised
		if supervised:
			self.include_sample_data('targets')
			self.dout = self.din


	def is_supervised(self):
		return self._is_supervised


	def get_targets(self, idx=None, **kwargs):
		return self._default_get('targets', idx=idx, **kwargs)


	def get_target_names(self):
		return self._target_names
	def get_target_space(self):
		return self._target_space
	def get_target_size(self):
		return len(self.get_target_space())



class Disentanglement(Supervised):
	_all_label_class_names = None
	_all_label_names = None
	_full_label_space = None


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.register_data_aliases('labels', 'targets')


	def generate_observations(self, labels):
		raise NotImplementedError


	def get_labels(self, idx=None, **kwargs):
		return self._default_get('labels', idx=idx, **kwargs)


	def get_target_names(self):
		names = super().get_target_names()
		if names is None:
			return self.get_label_names()
		return names
	def get_target_space(self):
		space = super().get_target_space()
		if space is None:
			return self.get_label_space()
		return space


	def get_label_names(self):
		return self._all_label_names
	def get_label_space(self):
		return self._full_label_space
	def get_label_sizes(self):
		return [dim.expanded_len() for dim in self.get_label_space()]


	def get_label_class_names(self, factor):
		if isinstance(factor, str):
			return self.get_label_class_names(self.get_label_names().index(factor))
		return self._all_label_class_names[factor]
	def get_label_class_space(self, factor):
		if isinstance(factor, str):
			return self.get_label_class_space(self.get_label_names().index(factor))
		return self.get_label_space()[factor]



class TopologicalBase(Disentanglement):
	_all_mechanism_class_names = None
	_all_mechanism_names = None
	_full_mechanism_space = None

	def __init__(self, *args, use_mechanisms=False, **kwargs):
		super().__init__(*args, **kwargs)
		self._use_mechanisms = use_mechanisms


	def get_label_space(self):
		if not self._use_mechanisms or self._full_mechanism_space is None:
			return super().get_label_names()
		return self._full_mechanism_space


	def get_mechanism_names(self):
		if self._all_mechanism_names is None:
			return self.get_label_names()
		return self._all_mechanism_names
	def get_mechanism_space(self):
		if self._full_mechanism_space is None:
			return self.get_label_space()
		return self._full_mechanism_space
	def get_mechanism_sizes(self):
		return [dim.expanded_len() for dim in self.get_mechanism_space()]


	def get_mechanism_class_names(self, mechanism):
		if isinstance(mechanism, str):
			return self.get_mechanism_class_names(self.get_mechanism_names().index(mechanism))
		if self._all_mechanism_class_names is not None:
			return self._all_mechanism_class_names[mechanism]
	def get_mechanism_class_space(self, mechanism):
		if isinstance(mechanism, str):
			return self.get_mechanism_class_space(self.get_mechanism_names().index(mechanism))
		return self.get_mechanism_space()[mechanism]



class ISupervisedDataset(Supervised):
	def __init__(self, observations=None, targets=None,
	             target_space=None, target_names=None,
	             din=None, dout=None, **kwargs):
		if din is None:
			din = tuple(observations.shape[1:])
		if dout is None:
			dout = tuple(labels.shape[1:])
		super().__init__(din=din, dout=dout)

		self.register_buffer('observations', observations)
		self.register_buffer('targets', targets)

		self._full_target_space = target_space
		self._all_target_names = target_names



# region Configurable Dataset

class Dataset(util.Dimensions, util.Configurable, DatasetBase):
	pass



class Deviced(util.Deviced, Dataset, DevicedBase): # Full dataset is in memory, so it can be moved to GPU
	pass



class Topological(Dataset, TopologicalBase):
	def __init__(self, A, use_mechanisms=None, **kwargs):
		if use_mechanisms is None:
			use_mechanisms = A.pull('use-mechanisms', False)
		super().__init__(A, use_mechanisms=use_mechanisms, **kwargs)



class Sourced(Dataset):
	def __init__(self, A, dataroot=None, **kwargs):
		super().__init__(A, **kwargs)
		self.root = util.get_data_dir(A) if dataroot is None else dataroot
	
	def get_root(self):
		return self.root



class Downloadable(Sourced):
	@classmethod
	def download(cls, A, **kwargs):
		raise NotImplementedError




class MissingFIDStatsError(Exception):
	def __init__(self, root, dim, modes, available=None):
		msg = f'no fid stats (dim={dim}, modes={modes}) found in: {root}'
		if available is not None:
			msg = msg + f' - available: {available}'
		super().__init__(msg)
		self.root = root
		self.modes = modes
		self.dim = dim
		self.available = available



class ImageDataset(Dataset, Observation):
	def __init__(self, A, root=None, fid_ident=unspecified_argument, **other):
		if fid_ident is unspecified_argument:
			fid_ident = A.pull('fid_ident', None)
		super().__init__(A, **other)
		self.register_data('images')
		self.register_data_aliases('images', 'observations')
		self.fid_ident = fid_ident


	def get_available_fid(self, name=None):
		if name is None:
			name = 'fid_stats.h5' if self.fid_ident is None else f'{self.fid_ident}_fid_stats.h5'
		if self.root is not None:
			path = self.root / name
			if path.is_file():
				with hf.File(path, 'r') as f:
					available = list(f.keys())

				available = [a.split('_')[:-1] for a in available if 'mu' in a]
				out = []
				for a in available:
					try:
						m, d = a
						out.append((int(d), m))
					except:
						pass
				return out


	def get_fid_stats(self, dim, *modes, name=None):
		if name is None:
			name = 'fid_stats.h5' if self.fid_ident is None else f'{self.fid_ident}_fid_stats.h5'
		available = None
		if self.root is not None:
			path = self.root / name
			if path.is_file():
				with hf.File(path, 'r') as f:
					available = f.keys()
					for mode in modes:
						key = f'{mode}_{dim}'
						if f'{key}_mu' in f:
							return f[f'{key}_mu'][()], f[f'{key}_sigma'][()]

		raise MissingFIDStatsError(self.root, dim, modes, available)


	def get_images(self, idx=None, **kwargs):
		return self._default_get('images', idx=idx, **kwargs)


	def __len__(self):
		return len(self.images)



class MissingDatasetError(Exception):
	def __init__(self, name):
		super().__init__(f'{name} (it can be downloaded using the "download-dataset" script)')



class ListDataset(Dataset):
	def __init__(self, ls):
		self.data = ls


	def __getitem__(self, idx):
		return self.data[idx]


	def __len__(self):
		return len(self.data)
	
	
	
# region Wrappers





# endregion
