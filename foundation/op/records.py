import sys, os
from pathlib import Path
import random
from omnibelt import get_now, save_yaml, save_json, load_json, load_yaml

import omnifig as fig

from .. import util

@fig.Component('records')
class Records(util.StatsClient, dict):
	
	def __init__(self, A):
		super().__init__()
		
		self.rng = random.Random()
		
		self.update(self._init_info(A))
		self.stats = self._init_stats(A)
		
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
			'mode': A.pull('mode', 'train'),
			'val': 0,
			
			'seed': seed,
			'epoch_seed': self._increment_rng(A.pull('epoch_seed', seed)),
			
			'history': A.pull('_history', None, silent=False),
			'argv': sys.argv,
			'timestamp': get_now(),
		}
		
		track_best = A.pull('track_best', False)
		
		if track_best:
			info['best'] = {'loss': None, 'checkpoint': None}
		
		return info
	
	def _init_stats(self, A=None):
		if A is not None:
			if 'stats' not in A:
				A.push('stats._type', 'stats-manager', silent=True, force_root=True)
			stats = A.pull('stats')
			self._active_stats = {self['mode']:stats}
			self.archive = {}
			return stats
		
	def get_description(self): # for logging
		
		if bar is not None:
			bar.update(1)
			title = '{} ({})'.format(mode, records['total_epochs'][mode] + 1) \
				if mode in records['total_epochs'] else mode
			loss_info = ' Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(),
			                                            stats['loss'].smooth.item()) \
				if stats['loss'].count > 0 else ''
			bar.set_description('{} ckpt={}{}'.format(title, records['checkpoint'], loss_info))
		
		elif self.print_time():
			
			total_steps = self.get_total_steps()
			title = '{} ({})'.format(mode, records['total_epochs'][mode] + 1) \
				if mode in records['total_epochs'] else mode
			loss_info = 'Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(),
			                                           stats['loss'].smooth.item()) \
				if stats['loss'].count > 0 else ''
			
			tm = time.strftime("%H:%M:%S")
			print(f'[ {tm} ] {title} {total_steps}/{step_limit} {loss_info}')
			
			sys.stdout.flush()
		raise NotImplementedError # TODO
		
	def get_short_description(self): # for progress bar
		raise NotImplementedError # TODO
	
	def register_stats_client(self, client, fmt=None):
		self.stats.register_client(client, fmt=fmt)
		
	def create_stats(self, mode=None):
		new = self.stats.copy()
		new.reset()
		if mode is not None:
			self._active_stats[mode] = new
		return new
	
	def prep(self):
		self.stats.pull_clients()
	
	def switch_mode(self, mode=None):
		self['mode'] = mode
		if mode not in self.stats:
			self.create_stats(mode)
		self.stats = self._active_stats[mode]
		self.stats.activate()
		
	def archive(self, mode=None, remove_from_active=True):
		if mode is None:
			mode = self['mode']
		if mode not in self.archive:
			self.archive[mode] = []
		if mode in self._active_stats:
			self.archive[mode].append(self._active_stats[mode].export())
			if remove_from_active:
				del self._active_stats[mode]
		
	def _increment_rng(self, seed):
		self.rng.seed(seed)
		return self.rng.getrandbits(32)
	
	def next_seed(self):
		self['epoch_seed'] = self._increment_rng(self['epoch_seed'])
		return self['epoch_seed']
	
	# region File I/O
	
	def _get_load_fn(self, path):
		return load_json if path.suffix == '.json' else load_yaml
	
	def import_info(self, path):
		path = Path(path)
		if path.is_dir():
			path = path / f'info.{self._info_format}'
		self.update(self._get_load_fn(path)(path))
	
	def import_stats(self, path):
		path = Path(path)
		if path.is_dir():
			path = path / f'stats.{self._stats_format}'
		self.archive.update(self._get_load_fn(path)(path))
	
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
		return self._get_save_fn(path)(self.archive, path)

	def export_info(self, path):
		path = Path(path)
		if path.is_dir():
			path = path / f'stats.{self._stats_format}'
		return self._get_save_fn(path)(dict(self.items()), path)
	
	def export(self, path, info=True, stats=True):
		if info:
			self.import_info(path)
		if stats:
			self.import_stats(path)
	
	# endregion
	
	pass



