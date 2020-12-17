import sys, os
from pathlib import Path
import time
import socket
import random

from collections import OrderedDict

import humpack as hp

import omnifig as fig
from omnifig.errors import MissingParameterError

from omnibelt import load_yaml, save_yaml, get_now, create_dir



FD_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(FD_PATH), 'trained_nets')


@fig.Component('clock')
class Clock:
	def __init__(self, A, **kwargs):
		self._ticks = 0
		self._alerts = OrderedDict()
		self._info = None
	
	def get_info(self):
		return self._info
	
	def set_info(self, info):
		self._info = info
	
	def register_alert(self, name, alert, **unused):
		if name is None:
			name = f'{alert}#{id(alert)}'
		self._alerts[name] = alert
	
	def clear(self):
		self._alerts.clear()
		self._info = None
	
	def _call_alert(self, name, alert, info=None):
		if info is None:
			info = self._info
		return alert.activate(self._ticks, info)
	
	def get_time(self):
		return self._ticks
	
	def __len__(self):
		return self.get_time()
	
	def set_time(self, ticks):
		self._ticks = ticks
	
	def tick(self, info=None):
		for name, alert in self._alerts.items():
			if alert.check(self._ticks, info):
				self._call_alert(name, alert, info=info)
		self._ticks += 1
	
	def step(self, info=None, n=None):
		if n is None:
			n = 1
		for _ in range(n):
			self.tick(info=info)

class Alert:
	def __init__(self, A=None, **kwargs):
		pass
	
	def check(self, tick, info=None):
		return True
	
	def activate(self, tick, info=None):
		'''

		:param tick: int
		:param info: object passed to clock.tick()
		:return: new value (representative of the alert) or none
		'''
		pass

class CustomAlert(Alert):
	def __init__(self, activate=None, check=None, **kwargs):
		super().__init__(**kwargs)
		self._activate = activate
		self._check = check
	
	def check(self, tick, info=None):
		if self._check is None:
			return True
		return self._check(tick, info=info)
	
	def activate(self, tick, info=None):
		if self._activate is None:
			pass
		return self._activate(tick, info=info)


class RunNotFoundError(Exception):
	def __init__(self, name):
		super().__init__(f'Run not found: {name}')


def wrap_script(script_name, A, **kwargs):
	A.begin()
	
	obj = fig.run(script_name, A, **kwargs)
	
	store_keys = A.pull('store_keys', [])
	
	stored = {}
	for key in store_keys:
		try:
			val = A.pull(key, silent=True)
		except MissingParameterError:
			pass
		else:
			stored[key] = val
	
	A.abort()
	
	for key, val in stored.items():
		A.push(key, val, silent=True, force_root=True)
	
	return obj

class Validation_Flag(Exception):
	pass

class Checkpoint_Flag(Exception):
	pass

class NoOverwriteError(Exception):
	pass

@fig.Script('load-run', description='Load a new or existing run') # the script just loads a run
def load_run(A):
	try:
		A = A.sub('run')
	except fig.MissingParameterError:
		pass

	A.push('_type', 'run', overwrite=False)

	return A.pull_self()


	

@fig.Component('run')
class Run(Alert):
	'''
	Holds all the data and functions to load, save, train, and evaluate runs.
	Runs include the model, datasets, dataloaders, the logger, and stats
	'''
	def __init__(self, A, silent=False):
		super().__init__(A)
		self.silent = A.pull('silent', silent)
		self.invisible = A.pull('invisible', False)
		
		A = A.get_root()
		A = self._find_origin(A)
		
		self._prep(A)
		
		if not self.invisible:
			self._setup_storage(A)
		
		self.purge() # reset payload objs
	
	# region Setup
	
	def __repr__(self):
		return f'RUN:{self.get_name()}'
	
	def __str__(self):
		return self.get_name()
	
	def _prep(self, A):
		if 'clock' not in A:
			A.push('clock._type', 'clock', overwrite=False, silent=True)
		self.clock = A.pull('clock', ref=True)

		A.push('records._type', 'records', overwrite=False, silent=True)

		self.A = A
	
	def purge(self): # remove any potentially large objects to free memory
		self.dataset = None
		self.model = None
		self.records = None
		
		self.loader = None
		self._loader = None
		
		self.batch = None
		self.out = None
		
		self.clock.clear()
		
		self.results = {}
	
	def _find_origin(self, A):
		
		path = A.pull('path', '<>resume', '<>load', None)
		novel = A.push('novel', path is None, overwrite=False)
		override = A.pull('override', {})
		
		if path is not None:
			path = Path(path)
			
			if not path.is_dir():
				root = A.pull('saveroot', '<>root', os.environ.get('FOUNDATION_SAVE_DIR', None))
				if root is None:
					raise RunNotFoundError(path)
				path = root / path
				if not path.is_dir():
					raise RunNotFoundError(path)
			
			config_path = path / 'config.yaml'
			
			if not self.silent:
				print(f'Loading Config: {config_path}')
			
			load_A = fig.get_config(path)
			if novel:
				load_A.update(A)
			A.clear()
			A.update(load_A)
			A.update(override)
		
		self.path = path
		self.novel = novel
		
		return A
		
	def _setup_storage(self, A):
		if self.path is None:
			self._create_storage(A)
		else:
			self._load_storage(A)
		
	def _create_storage(self, A):
		
		name = A.pull('run.name', '<>name', None)
		if name is None:
			name = self._gen_name(A)
			A.push('name', name)
		
		logdate = A.pull('name-include-date', True)
		if logdate:
			now = get_now()
			name = f'{name}_{now}'
		
		self.name = name
		
		path = None
		
		if not self.invisible:
			saveroot = A.pull('saveroot', os.environ.get('FOUNDATION_SAVE_DIR', DEFAULT_SAVE_PATH))
			saveroot = Path(saveroot)

			path = saveroot / self.name
			
			unique = A.pull('unique-name', True)
			if unique and self.novel and path.is_dir():
				idx = 2
				nameroot = self.name
				while path.is_dir():
					name = f'{nameroot}_{idx}'
					path = saveroot / name
				self.name = name
			
			path.mkdir(exist_ok=True)
			if self.novel:
				A.push('path', str(path))
		
		self.path = path
	
	def _load_storage(self, A):
		
		self.name = self.path.stem

		num = A.pull('ckpt-num', None, silent=self.silent)
		best = A.pull('best', False, silent=self.silent)
		if best:
			raise NotImplementedError
		last = A.pull('last', not best and num is None, silent=self.silent)
		
		ckpts = {int(path.stem[4:]):path for path in self.path.glob('ckpt*') if path.is_dir()}
		nums = sorted(ckpts.keys())
		
		if not len(nums):
			return
		
		ckpt = ckpts[nums[-1]] if last or num is None or num not in ckpts else ckpts[num]
		
		
		self.get_records().import_(ckpt)
		self.clock.set_time()
		
		ckpt = str(ckpt)
		A.push('dataset._load-ckpt', ckpt, overwrite=True, silent=self.silent)
		A.push('model._load-ckpt', ckpt, overwrite=True, silent=self.silent)
		A.push('records._load-ckpt', ckpt, overwrite=True, silent=self.silent)
	
	# endregion

	def _gen_name(self, A):
		return 'test'
	
	# def create_logger(self, path=None):
	#
	# 	A = self.get_config()
	#
	# 	if path is None:
	# 		path = self.path
	#
	# 	if path is None:
	# 		return None
	#
	# 	A.push('logger._type', 'logger', overwrite=False, silent=self.silent)
	# 	A.push('logger.log_dir', str(path), overwrite=True, silent=self.silent)
	#
	# 	logger = A.pull('logger', silent=self.silent)
	# 	return logger
	
	def create_dataset(self, A=None, **meta):
		if A is None:
			A = self.get_config().sub('dataset')
		return wrap_script('load-data', A, **meta)
	def create_model(self, A=None, **meta):
		if A is None:
			A = self.get_config().sub('model')
		return wrap_script('load-model', A, **meta)
	def create_records(self, A=None, **meta):
		if A is None:
			A = self.get_config().sub('records')
		return A.pull_self()
	
	def create_loader(self, mode='train', **kwargs):
		dataset = self.get_dataset()
		dataset.switch_to(mode)
		return dataset.to_loader(**kwargs)
		
	def create_results(self, A=None, **meta):
		raise NotImplementedError # TODO
	
	# region "Payload" objects - not loaded automatically

	def get_config(self):
		return self.A
	
	def get_clock(self):
		return self.clock

	def get_dataset(self, **meta):
		if self.dataset is None:
			self.dataset = self.create_dataset(**meta)
		return self.dataset
	def get_model(self, **meta):
		if self.model is None:
			self.model = self.create_model(**meta)
		return self.model
	def get_records(self, **meta):
		if self.records is None:
			self.records = self.create_records(**meta)
		return self.records
	
	def get_loader(self, activate=False, **kwargs):
		if self.loader is None:
			self.loader = self.create_loader(**kwargs)
		if activate and self._loader is None:
			self._loader = iter(self.loader)
		return self.loader
		
	
	def get_results(self, ident, remove_ext=True): # you must specify which results (ident used when results were created)
		
		if ident in self.results:
			return self.results[ident]

		fixed = ident.split('.')[0] if remove_ext else ident

		if fixed in self.results:
			return self.results[fixed]

		self.results[fixed] = self._load_results(ident)

		return self.results[fixed]

	# endregion
	
	# region Signals

	def get_path(self):
		return self.path

	def get_output(self):
		raise NotImplementedError

	# endregion
	
	# region Training Phases
	
	def activate(self, tick, info=None):
		return self.train_step()
	
	def continuous(self):
		
		self.startup()
		
		self.clock.step(info=self)
		
	
	def startup(self):
		
		A = self.get_config()
		clock = self.get_clock()

		clock.register_alert('data', CustomAlert(self.dataset_step))
		clock.register_alert('train', CustomAlert(self.train_step))

		dataset = self.get_dataset()
		model = self.get_model()
		records = self.get_records()
		
		validation = A.pull('validation', None)
		if validation is None:
			print('No validation')
		else:
			clock.register_alert('val', validation)
		
		
		if isinstance(records, Alert):
			clock.register_alert('log', records)

		A.push('vizualization._type', 'run/viz', overwrite=False)
		viz_step = A.pull('vizualization', None)
		if viz_step is None:
			clock.register_alert('viz', viz_step)
		
		if self.invisible:
			print('No checkpointing')
		else:
			A.push('checkpoint._type', 'run/checkpoint', overwrite=False, silent=True)
			checkpointer = A.pull('checkpoint', None)
		
			if checkpointer is None:
				print('No checkpointer found')
			else:
				clock.register_alert('checkpoint', checkpointer)
		
		path = self.get_path()
		if path is not None:
			config_path = self.get_path() / 'config.yaml'
			if not config_path.is_file():
				A.export(config_path)
		
		dataset.prep(model=model, records=records)
		model.prep(dataset=dataset, records=records)
		records.prep(dataset=dataset, model=model)
		
		mode = 'train'
		
		dataset.switch_to(mode)
		model.switch_to(mode)
		records.switch_to(mode)
		
		self.loader = None
		self.get_loader(activate=True, mode=mode, infinite=True)
	
	def log_step(self, out, tag_fmt='{}/train', measure_time=True):
		Q = self.train_state
		logger = self.get_logger()
		train_stats = Q.train_stats
		
		if self.viz_criterion is not None:
			train_stats.update('loss-viz', self.viz_criterion(out).detach())
		
		logger.set_step(self.records['total_samples']['train'] if Q.display_samples else self.records['total_steps'])
		logger.set_tag_format(tag_fmt)
		
		display = train_stats.smooths() if Q.display_smoothed else train_stats.avgs()
		for k, v in display.items():
			logger.add('scalar', k, v)
		
	def dataset_step(self):

		if self._loader is None:
			self.startup()
			
		self.batch = None
		self.out = None
		self.batch = next(self._loader)
	
	def train_step(self):
		
		batch = self.get_batch()
		
		if batch is None:
			raise RuntimeError('No batch found')
		
		self.out = self.model.step(batch)
		
		
	def prep_eval(self):
		
		A = self.get_config()
		
		logger = self.get_logger()
		
		use_testset = A.pull('eval.use_testset', False)
		
		if not self.silent and use_testset:
			print('Using the testset')
		elif not self.silent:
			print('NOT using the testset (val instead)')
		
		total_steps = self.get_total_steps()
		
		if not self.silent:
			print(f'Loaded best model, trained for {total_steps} iterations')

		self.eval_identifier = A.push('eval.identifier', 'eval')

		logger.set_step(total_steps)
		logger.set_tag_format('{}/{}'.format(self.eval_identifier, '{}'))

		root = self.get_save_path()
		assert root is not None, 'Apparently no save_dir was found'
		
		self.results_path = os.path.join(root, f'{self.eval_identifier}.pth.tar')
		overwrite = A.pull('eval.overwrite', False)
		if not self.silent and os.path.isfile(self.results_path) and not overwrite:
			print('WARNING: will not overwrite results, so skipping evaluation')
			raise NoOverwriteError

		self.eval_mode = A.push('eval.mode', 'test' if use_testset else 'val', overwrite=False)

		self.get_datasets(self.eval_mode)
		datasets = self.get_datasets()
		self.get_model().prep(datasets)

		# self.eval_dataloader = None
		
	def evaluate(self, mode=None, dataloader=None):

		if mode is None:
			mode = self.eval_mode
		if dataloader is None:
			dataloader = self.get_loaders(mode)
		if dataloader is None:
			dataloader = self.get_loaders('train')

		model = self.get_model()

		model.eval()

		results = model.evaluate(dataloader, logger=self.get_logger(),
		                         A=self.get_config().sub('eval'), run=self)

		if results is not None:
			ident = self.eval_identifier
			self.save_results(ident, results)

		return results
		
	def exit_run(self, cause, code=None):
		
		cmsg = f' (code={code})'
		
		if not self.silent:
			print(f'Exiting due to {cause}{cmsg}')
		
		if code is not None:
			sys.exit(code)
	
	# endregion

@fig.AutoModifier('extendable')
class Extendable(Run):
	
	def _find_origin(self, A):
		A = super()._find_origin(A)
		
		extend = A.pull('extend', None)
		if extend is not None:
			A.push('training.step_limit', extend)
		
		return A

		
@fig.AutoModifier('timed-run')
class Timed(Run):
	
	def startup(self):
		super().startup()
		
		A = self.get_config()
		
		time_limit = A.pull('training.time_limit', None)
		if time_limit is not None:
			if not self.silent:
				print(f'SOFT TIME LIMIT: {time_limit:2.2f} hrs')
			self.train_state.start_time = time.time()
			time_limit = 3600 * time_limit
		
			self.train_state.time_limit_exit_code = A.pull('time_limit_exit_code', 3)
		
		self.train_state.time_limit = time_limit

	def save_checkpoint(self, save_model=True):
		super().save_checkpoint(save_model=save_model)
		
		time_limit = self.train_state.time_limit
		if time_limit is not None:
			start_time = self.train_state.start_time
			if (time.time() - start_time) > time_limit:
				self.exit_run('timelimit reached', code=self.train_state.time_limit_exit_code)
			

		