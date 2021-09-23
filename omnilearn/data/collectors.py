
import sys, os
from pathlib import Path
from collections import OrderedDict
from wrapt import ObjectProxy
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset, TensorDataset
from torch.utils.data.dataloader import default_collate as pytorch_collate
import h5py as hf

from omnibelt import unspecified_argument, InitWall

# from .collate import SampleFormat

from .. import util



# region DatasetBases

class MissingDataError(Exception):
	def __init__(self, *names):
		super().__init__('Missing data: {}'.format(', '.join(names)))
		self.names = names


class DatasetBase(util.DimensionBase, InitWall, PytorchDataset):
	_available_modes = {'train'}
	_available_data = {}
	_sample_format = None

	def __init_subclass__(cls, available_data={}, available_modes=[],
	                      sample_format=None, sample_format_type=None, **kwargs):
		super().__init_subclass__(**kwargs)

		if sample_format is None:
			sample_format = []
		if cls._sample_format is None:
			cls._sample_format = sample_format
		cls._available_modes = {*available_modes, *cls._available_modes}
		cls._available_data =  {**available_data, **cls._available_data}
		for key in cls._sample_format:
			if key not in cls._available_data:
				cls._available_data[key] = None
		cls._sample_format_type = sample_format_type


	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._available_modes = self.__class__._available_modes.copy()
		self._available_data = self.__class__._available_data.copy()
		self._sample_format = self.__class__._sample_format.copy()


	def get(self, name=None, idx=None, **kwargs):
		if name is not None:
			alias = name
			while alias is not None and name in self._available_data:
				name = alias
				getter = getattr(self, f'get_{name}', None)
				if getter is None:
					data = getattr(self, name, None) # check for tensor
					if data is not None:
						getter = lambda idx, **_: data[idx]
				if getter is not None:
					return getter(idx, **kwargs)
				alias = self._available_data[name]
			raise MissingDataError(name)

		needed = self._sample_format
		if needed is None or len(needed) == 0:
			needed = list(key for key in self._available_data if self._available_data[key] is None)
		return self.format({name: self.get(name=name, idx=idx) for name in needed})


	def __len__(self):
		for key in self._sample_format:
			data = getattr(self, key, None)
			if data is not None:
				return len(data)


	def __getitem__(self, item):
		return self.get(idx=item)


	# def __getattr__(self, item):
	# 	try:
	# 		return super().__getattribute__(item)
	# 	except AttributeError:
	# 		alias = super().__getattribute__('_available_data_aliases').get(item, None)
	# 		if aslias is not None:
	# 			return super().__getattribute__(alias)
	# 		raise


	def register_data(self, name, data=None, strict=True): # either provide the tensor, or implement get_{data}
		if data is not None:
			setattr(self, name, data)
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


	def format(self, sample):
		if self._sample_format_type is not None:
			return self._sample_format_type(**sample)
		return tuple(sample[n] for n in self._sample_format)


	@classmethod
	def add_available_modes(cls, *modes):
		cls._available_modes.update(modes)


	@classmethod
	def get_available_modes(cls):
		return cls._available_modes



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
	_sample_format = ['observations']
	_full_observation_space = None


	# def get_observation(self, idx=None):
	# 	if isinstance(self, Batchable):
	# 		return self[torch.arange(len(self))][0]
	# 	return util.pytorch_collate([self[i][0] for i in range(len(self))])


	# def replace_observations(self, observations):
	# 	self._check_lock()
	# 	return self._replace_observations(observations)
	# def _replace_observations(self, observations):
	# 	raise NotImplementedError


	def get_observations(self, idx=None):
		if idx is None:
			return self.observations
		return self.observations[idx]


	def get_observation_space(self):
		return self._full_observation_space



class Supervised(Observation):
	_full_label_space = None
	_all_label_names = None

	def __init__(self, supervised=True, **kwargs):
		super().__init__(**kwargs)
		self._is_supervised = supervised
		if supervised:
			self.include_sample_data('targets')
			self.dout = self.din


	def is_supervised(self):
		return self._is_supervised


	def get_targets(self, idx=None):
		if idx is None:
			return self.targets
		return self.targets[idx]

	# def get_target(self):
	# 	if isinstance(self, Batchable):
	# 		return self[torch.arange(len(self))][1]
	# 	return util.pytorch_collate([self[i][1] for i in range(len(self))])


	# def replace_labels(self, labels):
	# 	self._check_lock()
	# 	return self._replace_labels(labels)
	# def _replace_labels(self, labels):
	# 	raise NotImplementedError


	def get_label_names(self):
		return self._all_label_names
	def get_label_space(self):
		return self._full_label_space



class Disentanglement(Supervised):
	_available_data = {'targets':'labels'}
	_all_mechanism_names = None
	_all_mechanism_class_names = None
	_full_mechanism_space = None


	def get_labels(self, idx=None):
		if idx is None:
			return self.labels
		return self.labels[idx]


	def get_mechanism_class_names(self, mechanism):
		if isinstance(mechanism, str):
			return self.get_mechanism_class_names(self.get_mechanism_names().index(mechanism))
		if self._all_mechanism_class_names is not None:
			return self._all_mechanism_class_names[mechanism]
	def get_mechanism_class_space(self, mechanism):
		if isinstance(mechanism, str):
			return self.get_mechanism_class_space(self.get_mechanism_names().index(mechanism))
		return self.get_mechanism_space()[mechanism]

	def get_mechanism_names(self):
		return self._all_mechanism_names
	def get_mechanism_space(self):
		if self._full_mechanism_space is None:
			return self.get_label_space()
		return self._full_mechanism_space

	def get_label_names(self):
		return self._all_mechanism_names
	def get_label_sizes(self):
		return list(map(len, self.get_mechanism_names()))



class ISupervisedDataset(Supervised):
	def __init__(self, observations=None, labels=None,
	             label_space=None, label_names=None,
	             din=None, dout=None, **kwargs):
		if din is None:
			din = tuple(observations.shape[1:])
		if dout is None:
			dout = tuple(labels.shape[1:])
		super().__init__(observations, labels, din=din, dout=dout)

		self.register_buffer('observations', observations)
		self.register_buffer('targets', labels)

		self._full_label_space = label_space
		self._all_label_names = label_names



# region Configurable Dataset

class Dataset(util.Dimensions, util.Configurable, DatasetBase):
	pass



class Deviced(util.Deviced, Dataset, DevicedBase): # Full dataset is in memory, so it can be moved to GPU
	pass



class Batchable(Deviced): # you can select using a full batch
	def allow_batched(self):
		return True



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



class ImageDataset(Dataset):
	def __init__(self, A, root=None, fid_ident=unspecified_argument, **other):
		if fid_ident is unspecified_argument:
			fid_ident = A.pull('fid_ident', None)
		super().__init__(A, **other)
		self.register_data_aliases('observations', 'images')
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

	def get_images(self, idx=None):
		if idx is None:
			return self.images
		return self.images[idx]

	def __len__(self):
		return len(self.images)


class MissingDatasetError(Exception):
	def __init__(self, name):
		super().__init__(f'Missing dataset {name} (it can be downloaded using the "download-dataset" script)')



class ListDataset(Dataset):
	def __init__(self, ls):
		self.data = ls


	def __getitem__(self, idx):
		return self.data[idx]


	def __len__(self):
		return len(self.data)
	
	
	
# region Wrappers





# endregion
