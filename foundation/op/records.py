import sys, os
import time
from pathlib import Path
import random
from datetime import datetime
from omnibelt import get_now, save_yaml, save_json, load_json, load_yaml

import omnifig as fig

from .clock import Freq

from .. import util
from ..util.features import Configurable, Switchable, Checkpointable, Seed


class SimpleRecords(Freq, Switchable, Seed, Configurable, dict):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		self.update(self._init_info(A))
		self.stats = self._init_stats(A)
		self.logger = self._init_logger(A)
		
		self._stats_format = A.pull('stats_format', 'json')
		self._info_format = A.pull('info_format', 'yaml')
		
		self._log_smooths = A.pull('log-smooth', True)
		
		self.purge()
		
	def purge(self):
		self._model = None
		self._dataset = None
		
	def prep(self, dataset=None, model=None):
		self._model = model
		self._dataset = dataset
	
	def _init_info(self, A):
		info = {
			'total_steps': {},
			'total_samples': {},
			'total_epochs': {},
			
			'checkpoint': None,
			
			'history': A.pull('_history', None, silent=False),
			'argv': sys.argv,
			'timestamp': datetime.time(),
		}
		
		return info
	
	def _init_stats(self, A=None):
		if 'stats' not in A:
			A.push('stats._type', 'stats-manager', silent=True, force_root=True, overwrite=False)
		stats = A.pull('stats')
		return stats
		
	def _init_logger(self, A=None):
		if 'logger' not in A:
			A.push('logger._type', 'logger', silent=True, overwrite=False)
		logger = A.pull('logger')
		return logger
		
	def activate(self, tick, info=None):
		self.step(tick)
		
	def step(self, num=None):
		
		if self.logger is not None:
			if num is not None:
				self.set_logger_step(num)
			
			display = self.stats.smooths() if self._log_smooths else self.stats.vals()
			
			for name, val in display.items():
				self.log('scalar', name, val)
		
		
	def set_logger_step(self, tick):
		if self.logger is not None:
			self.logger.set_step(tick)
		
	def log(self, data_type, tag, *args, global_step=None, **kwargs):
		if self.logger is not None:
			self.logger.add(data_type, tag, *args, global_step=global_step, **kwargs)
		
	def switch_to(self, mode='train'):
		super().switch_to(mode)
		if self.stats is not None:
			self.stats.switch_to(mode)
		
	def prep_checkpoint(self, tick):
		
		self['checkpoint'] = tick
	
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
		self.stats.load(self._get_load_fn(path)(path))

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
			path = path / f'stats.{self._stats_format}'
		return self._get_save_fn(path)(self.stats.export(), path)

	def export_info(self, path):
		path = Path(path)
		if path.is_dir():
			path = path / f'info.{self._info_format}'
		return self._get_save_fn(path)(dict(self.items()), path)

	def export(self, path, info=True, stats=True):
		if info:
			self.export_info(path)
		if stats:
			self.export_stats(path)
	

class Checkpointed(Checkpointable, SimpleRecords):

	def __init__(self, A, **kwargs):
		
		ckpt = A.pull('_load-ckpt', None, silent=True)
		
		super().__init__(A, **kwargs)
		
		if ckpt is not None:
			self.load_checkpoint(ckpt)
	
	def checkpoint(self, path, ident=None):
		self.export(path)
		
	def load_checkpoint(self, path, ident=None):
		self.import_(path)


@fig.Component('records')
class Records(Checkpointed, SimpleRecords):
	pass


