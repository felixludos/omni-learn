
import sys, os
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset
import h5py as hf

import omnifig as fig

from .. import util

from .register import dataset_registry
from .collectors import Shuffle_Dataset, Subset_Dataset
from .loaders import Featured_DataLoader, BatchedDataLoader

class DataManagerBase(util.Dimensions, util.Switchable, util.Deviced):
	def __init__(self, A, **kwargs):
		
		name = A.pull('_dataset_type', '<>ident', '<>name')
		mods = A.pull('_dataset_mod', [])
		
		default_mode = A.pull('default_mode', '<>mode', 'train')
		A.push('mode', default_mode, overwrite=False, silent=True)
		
		skip_load = A.pull('skip_load', False)
		
		super().__init__(A, **kwargs)
		self._default_mode = default_mode
		
		self.A = A
		
		cmpn_name = dataset_registry.get(name, None)
		if cmpn_name is None:
			raise NotImplementedError
		self.A.push('_type', cmpn_name, silent=True)
		self.A.push('_mod', mods, silent=True)
		
		self.purge()
		if not skip_load:
			dataset = self.startup()
			if isinstance(dataset, util.Dimensions):
				self.din, self.dout = dataset.get_dims()
				A.push('din', self.din, silent=True)
				A.push('dout', self.dout, silent=True)
	
	def startup(self):
		self.switch_mode()
		return self.get_data()
		
	def purge(self):
		self._active = None
		self._modes = {}
	
	def _create_mode(self, mode):
		self.A.begin()
		if self.A.contains_nodefault(mode):
			self.A.update(self.A.sub(mode))
		self.A.push('mode', mode, silent=True)
		dataset = self.A.pull_self()
		self.A.abort()
		
		self._modes[mode] = dataset
		return dataset
	
	def switch_mode(self, mode=None):
		if mode is None:
			mode = self._default_mode
		
		super().switch_mode(mode)
		self._active = self._modes[mode] if mode in self._modes else self._create_mode(mode)
	
	def get_data(self, mode=None):
		if mode is not None:
			self.switch_mode(mode)
		return self._active
	
	def __len__(self):
		return len(self.get_data())
	
	def __getitem__(self, item):
		return self.get_data()[item]
	
	def __getattribute__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError as e:
			if self._active is not None:
				try:
					return self._active.__getattribute__(item)
				except AttributeError:
					raise e
			else:
				raise e

@fig.AutoModifier('datamanager/loadable')
class Loadable(DataManagerBase):
	def __init__(self, A, **kwargs):
		num_workers = A.pull('num_workers', 0)
		batch_size = A.pull('batch_size', 64)
		shuffle = A.pull('shuffle', True)
		drop_last = A.pull('drop_last', False)
		loader_device = A.pull('step_device', '<>device', 'cuda' if torch.cuda.is_available() else 'cpu')
		infinite = A.pull('infinite', False)
		extractor = A.pull('extractor', None)
		allow_batched = A.pull('allow_batched', True)
		
		super().__init__(A, **kwargs)
		
		pin_memory = A.pull('pin_memory', self.device == 'cpu')
		
		self._loader_settings = {
			'num_workers': num_workers,
			'batch_size': batch_size,
			'shuffle': shuffle,
			'drop_last': drop_last,
			'device': loader_device,
			'pin_memory': pin_memory,
		}
		self._allow_batched = allow_batched
		self._infinite_loader = infinite
		self._loader_extractor = extractor
	
	def get_loader(self, mode=None, **kwargs):
		dataset = self.get_data(mode=mode)
		return self.to_loader(dataset, **kwargs)
	
	def to_loader(self, dataset, infinite=None, extractor=None, **updates):
		settings = self._loader_settings.copy()
		settings.update(updates)
		
		loader_cls = Featured_DataLoader
		
		if self._allow_batched:
			try:
				assert dataset.allow_batched()
			except (AttributeError, AssertionError):
				pass
			else:
				print('Using batched data loader')
				loader_cls = BatchedDataLoader
		
		loader = loader_cls(dataset, **settings)
		
		if infinite is None:
			infinite = self._infinite_loader
		if extractor is None:
			extractor = self._loader_extractor
		if infinite:
			return util.make_infinite(loader, extractor=extractor)
		return loader


@fig.AutoModifier('datamanager/splitable')
class Splitable(DataManagerBase):
	def __init__(self, A, **kwargs):
		
		split = A.pull('split', {})
		shuffle_split = A.pull('shuffle-split', True)
		split_src = A.pull('split-src', 'train')
		
		skip_load = A.pull('skip_load', False, silent=True)
		
		super().__init__(A, **kwargs)
		
		self.shuffle_split = shuffle_split
		self.split_info = split
		self.split_src = split_src
		
		if not skip_load:
			self._split_loaded()
		
	def _split_loaded(self):
		if self.split_src == self.get_mode():
			
			dataset = self.get_data()
			
			skip_modes = dataset.get_available_modes()
			
			self.split_info = {mode:ratio for mode, ratio in self.split_info.items() if mode not in skip_modes}
			
			modes, ratios = zip(*self.split_info.items())
			
			splits = self.split(dataset, *ratios, shuffle=self.shuffle_split)
			
			for mode, split in zip(modes, splits):
				self._modes[mode] = split
			self._modes[self.split_src] = splits[-1]
			self.switch_mode()
		
	def split(self, dataset, *ratios, shuffle=True):
		
		for r in ratios:
			assert 0 < r < 1, f'{r}'
		
		assert sum(ratios) < 1, 'nothing left for remainder'
		
		if shuffle:
			dataset = Shuffle_Dataset(dataset)
		
		parts = []
		idx = 0
		for r in ratios:
			n = int(np.round(len(dataset) * r))
			if n < 1:
				raise Exception(f'invalid ratio: {r} for dataset len {len(dataset)}')
			parts.append(Subset_Dataset(dataset, torch.arange(idx, idx+n)))
			idx += n
		parts.append(Subset_Dataset(dataset, torch.arange(idx, len(dataset))))
		
		return parts


@fig.Component('dataset')
@fig.Script('load-data', description='Load datasets')
class DataManager(Loadable, Splitable, DataManagerBase):
	def __init__(self, A, **kwargs):
		A.push('dataroot', os.environ.get('FOUNDATION_DATA_DIR', 'local_data'), overwrite=False)
		super().__init__(A, **kwargs)











