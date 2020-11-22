import sys, os
from pathlib import Path
import random
from omnibelt import get_now, save_yaml, save_json, load_json, load_yaml

import omnifig as fig

@fig.Component('records')
class Records:
	
	def __init__(self, A):
		
		self.rng = random.Random()
		
		self._info = self._init_info(A)
		self._stats = self._init_stats(A)
		self._last_stats = None
		
		self._stats_format = A.pull('stats_format', 'json')
		self._info_format = A.pull('info_format', 'yaml')
		
	
	def _init_info(self, A):
		seed = A.pull('seed', self.rng.getrandbits(32))
		
		info = {
			'total_steps': 0,
			
			'total_samples': {'train': 0, 'val': 0},
			'total_epochs': {'train': 0, 'val': 0},
			# 'stats': {'train': [], 'val': []},
			
			'batch': 0,
			'checkpoint': None,
			'val': 0,
			
			'seed': seed,
			'epoch_seed': self._increment_rng(A.pull('seed', seed)),
			
			'history': A.pull('_history', [], silent=False),
			'argv': sys.argv,
			'timestamp': get_now(),
		}
		
		track_best = A.pull('track_best', False)
		
		if track_best:
			info['best'] = {'loss': None, 'checkpoint': None}
		
		return info
	
	def _init_stats(self, A):
		return {}
	
	def _increment_rng(self, seed):
		self.rng.seed(seed)
		return self.rng.getrandbits(32)
	
	def __getitem__(self, item):
		return self._info[item]
	
	def get_stats(self):
		return self._last_stats
	
	def next_seed(self):
		self['epoch_seed'] = self._increment_rng(self['epoch_seed'])
		return self['epoch_seed']
	
	def append_stats(self, mode, stats):
		if mode not in self._stats:
			self._stats[mode] = []
		self._stats[mode].append(stats)
		self._last_stats = stats
	
	# region File I/O
	
	def _get_load_fn(self, path):
		return load_json if path.suffix == '.json' else load_yaml
	
	def import_info(self, path):
		path = Path(path)
		if path.is_dir():
			path = path / f'info.{self._info_format}'
		self._info.update(self._get_load_fn(path)(path))
	
	def import_stats(self, path):
		path = Path(path)
		if path.is_dir():
			path = path / f'stats.{self._stats_format}'
		self._stats.update(self._get_load_fn(path)(path))
	
	def import_(self, path, info=True, stats=True):
		if info:
			self.import_info(path)
		if stats:
			self.import_stats(path)
	
	def _get_save_fn(self, path):
		return save_json if path.suffix == '.json' else save_yaml
	
	def export_stats(self, path):
		path = Path(path)
		if path.is_dir():
			path = path / f'info.{self._info_format}'
		return self._get_save_fn(path)(self._stats, path)

	def export_info(self, path):
		path = Path(path)
		if path.is_dir():
			path = path / f'stats.{self._stats_format}'
		return self._get_save_fn(path)(self._info, path)
	
	def export(self, path, info=True, stats=True):
		if info:
			self.import_info(path)
		if stats:
			self.import_stats(path)
	
	# endregion
	
	pass



