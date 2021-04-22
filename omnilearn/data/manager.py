
import sys, os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset, DataLoader, RandomSampler
import h5py as hf

from omnibelt import save_yaml, load_yaml

import omnifig as fig

from .. import util
from ..util.features import Checkpointable

from .register import dataset_registry, DatasetNotFoundError
from .collectors import Shuffle_Dataset, Subset_Dataset
from .loaders import Featured_DataLoader, BatchedDataLoader

from ..op.clock import AlertBase

class SimpleDataManager(util.Seed, util.Switchable, util.Deviced):
	def __init__(self, A, mode='train', **kwargs):
		
		name = A.pull('_dataset_type', '<>dataset-name', '<>name')
		mods = A.pull('_dataset_mod', [])
		
		default_mode = A.pull('default_mode', '<>mode', mode)
		A.push('mode', default_mode, overwrite=False, silent=True)

		aliases = A.pull('mode-aliases', {})

		super().__init__(A, mode=mode, **kwargs)

		self._default_mode = default_mode
		
		self.A = A

		cmpn_name = dataset_registry.get(name, None)
		if cmpn_name is None:
			raise DatasetNotFoundError(name)
		self.A.push('_type', cmpn_name, silent=True)
		self.A.push('_mod', mods, silent=True)

		self._mode_aliases = aliases

		self.purge()
			
	def startup(self, A=None):
		mode = self._default_mode
		return self.get_data(mode)

	def purge(self):
		self._active = None
		self._modes = {}
	
	def available_modes(self):
		return list(self._modes.keys())
	
	def _create_mode(self, mode):

		# self.A.begin()
		if self.A.contains_nodefault(mode):
			self.A.update(self.A.sub(mode))
		self.A.push('mode', mode, silent=True)
		dataset = self.A.pull_self()
		# self.A.abort()
		
		self._modes[mode] = dataset
		return dataset

	def _find_active(self, mode):
		if mode in self._modes:
			return self._modes[mode]
		elif mode in self._mode_aliases:
			return self._find_active(self._mode_aliases[mode])
		return self._create_mode(mode)

	def switch_to(self, mode):
		if mode is None:
			mode = self._default_mode

		super().switch_to(mode)
		self._active = self._find_active(mode)
	def get_data(self, mode=None):
		if mode is not None:
			self.switch_to(mode)
		return self._active
	
	def __len__(self):
		return len(self.get_data())
	
	def __getitem__(self, item):
		return self.get_data()[item]
	
	def __getattribute__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError as e:
			if item != '_active' and self._active is not None:
				try:
					return self._active.__getattribute__(item)
				except AttributeError:
					raise e
			else:
				raise e

@fig.AutoModifier('datamanager/loadable')
class Loadable(SimpleDataManager):
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
	
	def get_batch_size(self):
		return self._loader_settings.get('batch_size', None)
	
	def to_loader(self, dataset, infinite=None, extractor=None, **updates):
		settings = self._loader_settings.copy()
		settings.update(updates)
		
		loader_cls = Featured_DataLoader
		# loader_cls = DataLoader
		
		# if self._allow_batched:
		# 	try:
		# 		assert dataset.allow_batched()
		# 	except (AttributeError, AssertionError):
		# 		pass
		# 	else:
		# 		# print('Using batched data loader')
		# 		loader_cls = BatchedDataLoader

		if len(dataset) > 1e9:
			generator = settings.get('generator', None)
			seed = settings.get('seed', None)
			shuffle = settings.get('shuffle', False)
			if shuffle:
				if generator is None and seed is not None:
					generator = torch.Generator()
					generator.manual_seed(seed)
					settings['generator'] = generator
				settings['sampler'] = RandomSampler(dataset, replacement=True, generator=generator)
				settings['shuffle'] = False

		loader = loader_cls(dataset, **settings)
		
		if infinite is None:
			infinite = self._infinite_loader
		if extractor is None:
			extractor = self._loader_extractor
		if infinite:
			return util.make_infinite(loader, extractor=extractor)
		return loader


@fig.AutoModifier('datamanager/splitable')
class Splitable(SimpleDataManager):
	def __init__(self, A, skip_load=None, **kwargs):
		
		split = A.pull('split', {})
		shuffle_split = A.pull('shuffle-split', True)
		split_src = A.pull('split-src', 'train')
		
		super().__init__(A, **kwargs)
		
		self.shuffle_split = shuffle_split
		self.split_info = split
		self.split_src = split_src
		self._split_done = False
		
	def register_mode(self, mode, subset):
		self._modes[mode] = subset
		
	def _split_load(self, dataset):
		
		if self.split_src == self.get_mode():
			
			skip_modes = dataset.get_available_modes()
			
			self.split_info = {mode: ratio for mode, ratio in self.split_info.items() if mode not in skip_modes}
			
			if len(self.split_info):
			
				modes, ratios = zip(*self.split_info.items())
				
				splits = self.split(dataset, *ratios, shuffle=self.shuffle_split)
				
				for mode, split in zip(modes, splits):
					self.register_mode(mode, split)
				self.register_mode(self.split_src, splits[-1])
				dataset = splits[-1]
				self.switch_to(self.split_src)
		
		return dataset
		
	def startup(self, A=None):
		dataset = super().startup(A=A)
		if not self._split_done:
			dataset = self._split_load(dataset)
			self._split_done = True
		return dataset
		
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
			part = Subset_Dataset(dataset, torch.arange(idx, idx+n))
			parts.append(part)
			idx += n
		last = Subset_Dataset(dataset, torch.arange(idx, len(dataset)))
		parts.append(last)
		
		return parts

class Active(Loadable, AlertBase):
	def __init__(self, A, **kwargs):

		super().__init__(A, **kwargs)
		self.rng = random.Random()
		self.epoch_seed = A.pull('epoch_seed', '<>seed', self.rng.getrandbits(32))
		
		self._loader = None
	
	def _increment_rng(self, seed):
		self.rng.seed(seed)
		return self.rng.getrandbits(32)
	
	def _start_epoch(self, seed=None, **loader_args):
		if seed is None:
			seed = self.epoch_seed
		loader = self.get_loader(seed=seed, **loader_args)
		assert len(loader)
		
		loader = iter(loader)
		self.epoch_seed = self._increment_rng(self.epoch_seed)
		
		return loader
	
	def _end_epoch(self):
		pass
	
	def __next__(self):
		return self.get_batch()
	
	def get_batch(self, force=True, **loader_args):
		
		if self._loader is None:
			self._loader = self._start_epoch(**loader_args)
		
		try:
			return next(self._loader)
		except StopIteration:
			self._end_epoch()
			if force:
				self._loader = None
				return self.get_batch(**loader_args)
			raise
	
	def switch_to(self, mode):
		if self._loader is not None and self.get_mode() != mode:
			self._loader = None
		return super().switch_to(mode)
	
	def activate(self, tick, info=None):
		info.receive_batch(self.get_batch())

class InfoManager(Checkpointable, Active):
	def __init__(self, A, **kwargs):
		
		ckpt = A.pull('_load-ckpt', None, silent=True)
		
		super().__init__(A, **kwargs)
		
		self.info = self._init_info(A)
		
		if ckpt is not None:
			self.load_checkpoint(ckpt)
			
		self.purge()
		
	def _init_info(self, A):
		
		info = {
			'batch': None,
			'epoch': None,
			
			'epoch_seed': self.epoch_seed,
		}
		
		return info
	
	def purge(self):
		super().purge()
		self._records = None
	
	def prep(self, order, info=None):
		super().prep(order, info=info)
		
		self._records = info.get_records()

	def _increment_rng(self, seed):
		epoch_seed = super()._increment_rng(seed)
		self.info['epoch_seed'] = epoch_seed
		return epoch_seed
	
	def _start_epoch(self, **loader_args):
		if self.info['batch'] is None:
			self.info['batch'] = 0
		if self.info['epoch'] is None:
			self.info['epoch'] = 0
		loader = super()._start_epoch(**loader_args)
		# loader.skip(self.info['batch'])
		return loader
	
	def _end_epoch(self):
		self.info['batch'] = 0
		self.info['epoch'] += 1
		
		if self._records is not None:
			self._records['total_epochs'][self.get_mode()] += 1
		
		super()._end_epoch()
	
	def get_batch(self, force=True, **loader_args):
		batch = super().get_batch(force=force, **loader_args)
		self.info['batch'] += 1
		# mode = self.get_mode()
		# self._records['total_steps'][mode] += 1
		# self._records['total_samples'][mode] += batch.size(0)
		return batch
	
	def checkpoint(self, path, ident='dataset'):
		path = Path(path)
		
		path = path / f'{ident}.yaml'
		save_yaml(self.info, str(path))
	
	def load_checkpoint(self, path, ident='dataset'):
		path = Path(path)
		
		path = path / f'{ident}.yaml'
		self.info = load_yaml(str(path))
		self.epoch_seed = self.info.get('epoch_seed', self.epoch_seed)
		


@fig.Script('load-data', description='Load datasets')
def load_data(A):

	info = A.pull('dataset', None, raw=True)
	if info is None:
		A.push('_type', 'dataset', silent=True)
	else:
		A = info

	return A.pull_self()


@fig.Component('dataset')
class DataManager(InfoManager, Splitable, SimpleDataManager):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		skip_load = A.pull('skip_load', False)
		if not skip_load:
			dataset = self.startup()
			try:
				dataset.get_dims()
				# self.din, self.dout = dataset.get_dims()
			except AttributeError:
				pass
			else:
				if A is None:
					A = self.A
				A.push('din', self.din, silent=True)
				A.push('dout', self.dout, silent=True)










