
import sys, os
from pathlib import Path
from wrapt import ObjectProxy
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset, TensorDataset
import h5py as hf

from omnibelt import unspecified_argument, InitWall

from .. import util


class ExistingModes(util.SwitchableBase):
	def __init_subclass__(cls, **kwargs):
		cls.available_modes = {'train'}
	
	@classmethod
	def add_existing_modes(cls, *modes):
		cls.available_modes.update(modes)
	
	@classmethod
	def get_available_modes(cls):
		return cls.available_modes



# region DatasetBases

class DatasetBase(ExistingModes, util.DimensionBase, InitWall, PytorchDataset):
	pass



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



class Observation(DatasetBase):
	_full_observation_space = None

	def get_observations(self):
		raise NotImplementedError
	def update_data(self, indices):
		raise NotImplementedError

	def _replace_observations(self, observations):
		raise NotImplementedError

	def get_observation_space(self):
		return self._full_observation_space



class Supervised(Observation):
	_full_label_space = None
	_all_label_names = None

	def get_labels(self):
		raise NotImplementedError

	def _replace_labels(self, labels):
		raise NotImplementedError

	def get_label_names(self):
		return self._all_label_names
	def get_label_space(self):
		return self._full_label_space



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
		self.register_buffer('labels', labels)

		self._full_label_space = label_space
		self._all_label_names = label_names

	def __len__(self):
		return self.observations.size(0)

	def __getitem__(self, item):
		return self.observations[item], self.labels[item]

	def get_observations(self):
		return self.observations

	def get_labels(self):
		return self.labels

	def _replace_labels(self, labels):
		self.labels = labels

	def _replace_observations(self, observations):
		self.observations = observations

	def update_data(self, indices):
		self._replace_observations(self.observations[indices])
		self._replace_labels(self.labels[indices])



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
