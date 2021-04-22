
import sys, os
from pathlib import Path
from wrapt import ObjectProxy
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset
import h5py as hf

from omnibelt import unspecified_argument

from .. import util


class ExistingModes(util.Switchable):
	def __init_subclass__(cls, **kwargs):
		cls.available_modes = {'train'}
	
	@classmethod
	def add_existing_modes(cls, *modes):
		cls.available_modes.update(modes)
	
	@classmethod
	def get_available_modes(cls):
		return cls.available_modes

class DatasetBase(util.DimensionBase, PytorchDataset):
	pass

class Dataset(ExistingModes, util.Dimensions, util.Configurable, DatasetBase):
	pass

class Sourced(Dataset):
	def __init__(self, A, dataroot=None, **kwargs):
		super().__init__(A, **kwargs)
		self.root = util.get_data_dir(A) if dataroot is None else dataroot
	
	def get_root(self):
		return self.root
	
class Batchable(Dataset): # you can select using a full batch
	def allow_batched(self):
		return True


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
		self.__setattr__(name, buffer)

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

class Deviced(util.Deviced, DevicedBase): # Full dataset is in memory, so it can be moved to GPU
	pass


class Downloadable(Sourced):
	@classmethod
	def download(cls, A, **kwargs):
		raise NotImplementedError


class MissingDatasetError(Exception):
	def __init__(self, name):
		super().__init__(f'Missing dataset {name} (it can be downloaded using the "download-dataset" script)')


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


class List_Dataset(Dataset):

	def __init__(self, ls):
		self.data = ls

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)
	
	
	
# region Wrappers

class DatasetWrapper(ObjectProxy):
	pass
	
class Indexed_Dataset(DatasetWrapper):
	def __getitem__(self, idx):
		return idx, self.__wrapped__[idx]


class Subset_Dataset(DatasetWrapper):

	def __init__(self, dataset, indices=None):
		super().__init__(dataset)
		self._self_indices = indices

		try:
			device = self.__wrapped__.get_device()
			if self._self_indices is not None:
				self._self_indices = self._self_indices.to(device)
		except AttributeError:
			pass

	def __getitem__(self, idx):
		return self.__wrapped__[idx] if self._self_indices is None else self.__wrapped__[self._self_indices[idx]]

	def __len__(self):
		return len(self.__wrapped__) if self._self_indices is None else len(self._self_indices)

class Repeat_Dataset(ObjectProxy):

	def __init__(self, dataset, factor):
		super().__init__(dataset)
		self._self_factor = factor
		self._self_num_real = len(dataset)
		self._self_total = self._self_factor * self._self_num_real
		print('Repeating dataset {} times'.format(factor))

	def __getitem__(self, idx):
		return self.__wrapped__[idx % self._self_num_real]

	def __len__(self):
		return self._self_total


class Format_Dataset(DatasetWrapper):

	def __init__(self, dataset, format_fn, format_args=None, include_original=False):
		super().__init__(dataset)

		self._self_format_fn = format_fn
		self._self_format_args = {} if format_args is None else format_args
		self._self_include_original = include_original

	def __getitem__(self, idx):

		sample = self.__wrapped__[idx]

		formatted = self._self_format_fn(sample, **self._self_format_args)

		if self._self_include_original:
			return formatted, sample

		return formatted

class Shuffle_Dataset(DatasetWrapper):

	def __init__(self, dataset):
		super().__init__(dataset)

		self._self_indices = torch.randperm(len(self.__wrapped__)).clone()

		try:
			device = self.__wrapped__.get_device()
			self._self_indices = self._self_indices.to(device).clone()
		except AttributeError:
			pass

	def __getitem__(self, idx):
		return self.__wrapped__[self._self_indices[idx]]

# endregion
