
import sys, os
from pathlib import Path
from wrapt import ObjectProxy
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset
import h5py as hf

from .loaders import get_loaders
from .. import util


class ExistingModes:
	def __init_subclass__(cls, **kwargs):
		cls.available_modes = {'train'}
	
	def add_existing_modes(self, *modes):
		self.available_modes.update(modes)
	
	def get_available_modes(self):
		return self.available_modes

class DatasetBase(ExistingModes, util.Dimensions, util.Configurable, PytorchDataset):
	pass
	
	
class Batchable(DatasetBase): # you can select using a full batch
	def allow_batched(self):
		return True


class Deviced(util.Deviced, DatasetBase): # Full dataset is in memory, so it can be moved to GPU
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





class Image_Dataset(DatasetBase):

	def __init__(self, A, root=None, **other):
		dataroot = A.pull('dataroot', '<>root', root)
		super().__init__(A, **other)
		self.root = Path(dataroot) if dataroot is not None else None

	def get_fid_stats(self, mode, dim):
		if self.root is not None:
			path = self.root / 'fid_stats.h5'
	
			if path.is_file():
				with hf.File(path, 'r') as f:
					key = f'{mode}_{dim}'
					if f'{key}_mu' in f:
						return f[f'{key}_mu'][()], f[f'{key}_sigma'][()]
					else:
						raise Exception(f'{key} not found: {str(f.keys())}')
	
			raise Exception(f'no fid stats file found in: {self.root}')


class List_Dataset(DatasetBase):

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
		self.indices = indices

		try:
			device = self.__wrapped__.get_device()
			if self.indices is not None:
				self.indices = self.indices.to(device)
		except AttributeError:
			pass

	def __getitem__(self, idx):
		return self.__wrapped__[idx] if self.indices is None else self.__wrapped__[self.indices[idx]]

	def __len__(self):
		return len(self.__wrapped__) if self.indices is None else len(self.indices)

class Repeat_Dataset(DatasetWrapper):

	def __init__(self, dataset, factor):
		super().__init__(dataset)
		self.factor = factor
		self.num_real = len(dataset)
		self.total = self.factor * self.num_real
		print('Repeating dataset {} times'.format(factor))

	def __getitem__(self, idx):
		return self.__wrapped__[idx % self.num_real]

	def __len__(self):
		return self.total


class Format_Dataset(DatasetWrapper):

	def __init__(self, dataset, format_fn, format_args=None, include_original=False):
		super().__init__(dataset)

		self.format_fn = format_fn
		self.format_args = {} if format_args is None else format_args
		self.include_original = include_original

	def __getitem__(self, idx):

		sample = self.__wrapped__[idx]

		formatted = self.format_fn(sample, **self.format_args)

		if self.include_original:
			return formatted, sample

		return formatted

class Shuffle_Dataset(DatasetWrapper):

	def __init__(self, dataset):
		super().__init__(dataset)

		self._shfl_indices = torch.randperm(len(self.__wrapped__)).clone()

		try:
			device = self.__wrapped__.get_device()
			self._shfl_indices = self._shfl_indices.to(device).clone()
		except AttributeError:
			pass

	def __getitem__(self, idx):
		return self.__wrapped__[self._shfl_indices[idx]]

# endregion
