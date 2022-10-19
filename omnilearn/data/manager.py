
import sys, os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset, DataLoader, RandomSampler
import h5py as hf

from omnibelt import save_yaml, load_yaml, unspecified_argument

import omnifig as fig

from .. import util
from ..util.features import Checkpointable

from .register import dataset_registry, DatasetNotFoundError
from .wrappers import wrap_dataset, wrapper_registry
from .loaders import Featured_DataLoader, BatchedDataLoader
from .collectors import DataLike

from ..op.clock import AlertBase

#@fig.Component('datamanager/simple-dataset')
class SimpleDataManager(util.Seed, util.Switchable, util.Deviced, DataLike):
	def __init__(self, A, mode='train', dataset_config=unspecified_argument,
	             aliases=None, default_mode=None, **kwargs):

		if default_mode is None:
			default_mode = A.pull('default-mode', '<>mode', mode)

		if dataset_config is unspecified_argument:

			name = A.pull('_dataset_type', '<>dataset-name', '<>name')
			mods = A.pull('_dataset_mod', [])

			A.push('mode', default_mode, overwrite=False, silent=True)

		if aliases is None:
			aliases = A.pull('mode-aliases', {})

		super().__init__(A, mode=mode, **kwargs)

		self._default_mode = default_mode
		self._dataset_type, self._dataset_mods = None, None

		if dataset_config is unspecified_argument:
			cmpn_name = dataset_registry.get(name, None)
			if cmpn_name is None:
				raise DatasetNotFoundError(name)

			dataset_config = A

			# dataset_config.push('_type', cmpn_name, silent=True)
			# dataset_config.push('_mod', mods, silent=True)

			self._dataset_type, self._dataset_mods = cmpn_name, mods

		self.dataset_config = dataset_config

		self._mode_aliases = aliases

		self._modes = {}
		self.purge()


	def startup(self, A=None):
		mode = self._default_mode
		return self.get_data(mode)


	def purge(self):
		self._active = None
		self._modes.clear()


	def get_loaded_modes(self):
		return list(self._modes.keys())

	def register_mode(self, name, mode):
		self._modes[name] = mode

	def _prep_new_mode(self, dataset):
		return dataset


	def _create_mode(self, mode):

		# self.A.begin()
		# if self.dataset_config.contains_nodefault(mode):
		# 	self.dataset_config.update(self.dataset_config.sub(mode))
		if self._dataset_type is not None:
			self.dataset_config.push('_type', self._dataset_type, silent=True)
		if self._dataset_mods is not None:
			self.dataset_config.push('_mod', self._dataset_mods, silent=True)
		self.dataset_config.push('mode', mode, silent=True)
		dataset = self.dataset_config.pull_self()
		dataset.prepare()
		# self.A.abort()
		dataset = self._prep_new_mode(dataset)

		return self._store_mode(mode, dataset)


	def _store_mode(self, mode, dataset):
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



#@fig.AutoModifier('datamanager/loadable')
class Loadable(SimpleDataManager):
	def __init__(self, A, **kwargs): # TODO arg defaults
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


	def to_loader(self, dataset=None, infinite=None, extractor=None, **updates):
		if dataset is None:
			dataset = self._active
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



#@fig.AutoModifier('datamanager/splitable')
class Splitable(SimpleDataManager):
	def __init__(self, A, skip_load=None, **kwargs):
		
		split = A.pull('split', {})
		shuffle_split = A.pull('shuffle-split', True)
		split_src = A.pull('split-src', 'train')
		
		super().__init__(A, **kwargs)
		
		self.shuffle_split = shuffle_split
		self.split_info = split
		self.split_src = split_src
		self._split_done = split is None


	def _split_load(self, dataset):
		
		if self.split_src == self.get_mode():
			
			skip_modes = dataset.get_available_modes()
			done_modes = self.get_loaded_modes()

			split_ratios = {mode: ratio for mode, ratio in self.split_info.items()
			                if mode not in skip_modes and mode not in done_modes}
			
			if len(split_ratios):
			
				modes, ratios = zip(*split_ratios.items())
				
				splits = self.split(dataset, *ratios, shuffle=self.shuffle_split)
				
				for mode, split in zip(modes, splits):
					self._store_mode(mode, split)
				self._store_mode(self.split_src, splits[-1])
				dataset = splits[-1]
				self.switch_to(self.split_src)
		
		return dataset


	def startup(self):
		dataset = super().startup()
		if not self._split_done:
			dataset = self._split_load(dataset)
			self._split_done = True
		return dataset


	def split(self, dataset, *ratios, shuffle=True):
		
		for r in ratios:
			assert 0 < r < 1, f'{r}'
		
		assert sum(ratios) < 1, 'nothing left for remainder'
		
		if shuffle:
			dataset = wrap_dataset('shuffled', dataset)
		
		parts = []
		idx = 0
		for r in ratios:
			n = int(np.round(len(dataset) * r))
			if n < 1:
				raise Exception(f'invalid ratio: {r} for dataset len {len(dataset)}')
			part = wrap_dataset('subset', dataset, torch.arange(idx, idx+n), update_data=False)
			parts.append(part)
			idx += n
		last = wrap_dataset('subset', dataset, torch.arange(idx, len(dataset)), update_data=False)
		parts.append(last)

		# # TESTING
		# last = wrap_dataset('subset', last, num=100)
		# last = last.to_loader(batch_size=32, format='observations')
		#
		# print(last)
		#
		# x = next(iter(last))
		#
		# print(x)
		# print(len(x))
		#
		# print(x.shape)

		return parts



#@fig.AutoModifier('datamanager/sharable')
class Sharable(SimpleDataManager):
	def __init__(self, A, _modes=None, _current_mode=None, **kwargs):
		super().__init__(A, **kwargs)

		if _modes is not None:
			self._modes = _modes.copy()

		if _current_mode is not None:
			self.switch_to(_current_mode)


	def duplicate(self, base_cls=None, loaded_modes=None, **kwargs):
		if base_cls is None:
			base_cls = self.__class__
		if loaded_modes is None:
			loaded_modes = self._modes
		return base_cls(self.dataset_config, _modes=loaded_modes, _current_mode=self.get_mode(), **kwargs)


	def get_all_loaded(self):
		return self._modes.copy()



#@fig.AutoModifier('datamanager/wrapable')
class Wrapable(Sharable):
	def __init__(self, A, wrappers=unspecified_argument, mode_wrappers=unspecified_argument, **kwargs):
		if wrappers is unspecified_argument:
			wrappers = A.pull('wrappers', [])

		if mode_wrappers is unspecified_argument:
			mode_wrappers = A.pull('mode-wrappers', {})

		super().__init__(A, **kwargs)

		self._data_wrappers = []
		if wrappers is not None and len(wrappers):
			for wrapper in wrappers:
				cls = wrapper['ident']
				del wrapper['ident']
				self.register_wrapper(cls, *wrapper.get('args', ()), **wrapper.get('kwargs', {}))
		self._mode_wrappers = mode_wrappers


	def startup(self):
		out = super().startup()
		for mode, dataset in self._modes.items():
			if mode in self._mode_wrappers:
				self._modes[mode] = self._wrap_dataset(self._modes[mode], [
					(wrapper['ident'], wrapper.get('args', ()), wrapper.get('kwargs', {}))
					for wrapper in self._mode_wrappers[mode]])

		self.switch_to(self.mode)

		return out


	def register_wrapper(self, wrapper, *args, **kwargs):
		wrapper_registry.find(wrapper)

		info = (wrapper, args, kwargs)
		self._data_wrappers.append(info)
		for mode, data in self._modes.items():
			self._modes[mode] = self._wrap_dataset(self._modes[mode], [info])
		self.switch_to(self.get_mode())


	def clear_wrappers(self):
		self._data_wrappers.clear()


	def _prep_new_mode(self, dataset):
		return self._wrap_dataset(dataset)


	def _wrap_dataset(self, dataset, wrappers=None):
		if wrappers is None:
			wrappers = self._data_wrappers
		for cls, args, kwargs in wrappers:
			dataset = wrap_dataset(cls, dataset, *args, **kwargs)
		return dataset



class Statistics(util.SmartResults, SimpleDataManager):

	def has_datafile(self, ident, root=None, **kwargs):
		return super().has_datafile(ident, root=(self.get_data().get_root() if root is None else root), **kwargs)


	def update_datafile(self, ident, data, root=None, **kwargs):
		return super().update_datafile(ident, data, root=(self.get_data().get_root() if root is None else root), **kwargs)


	def get_datafile(self, ident, root=None, **kwargs):
		return super().get_datafile(ident, root=(self.get_data().get_root() if root is None else root), **kwargs)


	@staticmethod
	def _check_stat_reqs(props, reqs):
		return props == reqs


	def _find_stat_ID(self, details, create=False):
		# name = details.get('name', details.get('ID', details.get('ident', None)))
		if 'mode' not in details:
			details['mode'] = self.get_mode()
		details.update(self.get_hparams())
		name = details.get('name', details.get('ID', details.get('ident', None)))
		mode = details['mode']
		ID = mode if name is None else f'{mode}-{name}'

		if not self.has_datafile('stats/table'):
			self.update_datafile('stats/table', {}, separate_dict=False)
		table = self.get_datafile('stats/table')
		if ID not in table:
			table[ID] = []

		for reqs, ident in table[ID]:
			if self._check_stat_reqs(details, reqs):
				return ident

		new = f'{ID}-{len(table[ID])}'
		if not create:
			raise FileNotFoundError(new)

		table[ID].append([details, new])
		self.update_datafile('stats/table', table, separate_dict=False)
		return new


	def save_stats(self, details, stats, overwrite=True):
		ID = self._find_stat_ID(details, create=True)
		return self.update_datafile(f'stats/{ID}', stats, overwrite=overwrite)


	def load_stats(self, details):
		ID = self._find_stat_ID(details, create=False)
		return self.get_datafile(f'stats/{ID}', skip_cache=True)




# class OldWrapable(Sharable):
# 	def __init__(self, A, wrappers=unspecified_argument, **kwargs):
# 		if wrappers is unspecified_argument:
# 			wrappers = A.pull('wrappers', [])
#
# 		super().__init__(A, **kwargs)
#
# 		self._data_wrappers = []
# 		if wrappers is not None and len(wrappers):
# 			for wrapper in wrappers:
# 				cls = wrapper['ident']
# 				del wrapper['ident']
# 				self.register_wrapper(cls, **wrapper)
#
#
# 	def register_wrapper(self, wrapper, args=(), kwargs={}, **resolve_kwargs):
# 		cls = resolve_wrappers(wrapper, **resolve_kwargs)
#
# 		info = (cls, args, kwargs)
# 		self._data_wrappers.append(info)
# 		for mode, data in self._modes.items():
# 			self._modes[mode] = self._wrap_dataset(data, [info])
# 		self.switch_to(self.get_mode())
#
#
# 	def clear_wrappers(self):
# 		self._data_wrappers.clear()
#
#
# 	def _store_mode(self, mode, dataset):
# 		return super()._store_mode(mode, self._wrap_dataset(dataset))
#
#
# 	def _wrap_dataset(self, dataset, wrappers=None):
# 		if wrappers is None:
# 			wrappers = self._data_wrappers
# 		for cls, args, kwargs in wrappers:
# 			dataset = cls(dataset, *args, **kwargs)
# 		return dataset



class Active(Loadable, AlertBase):
	def __init__(self, A, **kwargs):

		self._loader = None
		super().__init__(A, **kwargs)
		self.rng = random.Random()
		self.epoch_seed = A.pull('epoch_seed', '<>seed', self.rng.getrandbits(32))



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



#@fig.Script('load-data', description='Load datasets')
def load_data(A):

	info = A.pull('dataset', None, raw=True)
	if info is None:
		A.push('_type', 'dataset', silent=True)
	else:
		A = info

	dataset = A.pull_self()

	return dataset



#@fig.Component('dataset')
class DataManager(InfoManager, Wrapable, Splitable, Statistics, SimpleDataManager):
	def __init__(self, A, skip_load=None, **kwargs):
		super().__init__(A, **kwargs)

		if skip_load is None:
			skip_load = A.pull('skip_load', False)
		if not skip_load:
			dataset = self.startup()
			try:
				din, dout = dataset.get_dims()
				# self.din, self.dout = dataset.get_dims()
			except AttributeError:
				pass
			else:
				if A is None:
					A = self.dataset_config
				A.push('din', din, silent=True)
				A.push('dout', dout, silent=True)


	def duplicate(self, base_cls=None, loaded_modes=None, **kwargs):
		return super().duplicate(base_cls=base_cls, loaded_modes=loaded_modes, skip_load=True, **kwargs)







