import sys, os
from pathlib import Path
import time
import socket
import random

from collections import OrderedDict

import humpack as hp

import omnifig as fig
from omnifig import Configurable
from omnifig.errors import MissingParameterError

from omnibelt import load_yaml, save_yaml, get_now, create_dir



FD_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(FD_PATH), 'trained_nets')



class RunNotFoundError(Exception):
	def __init__(self, name):
		super().__init__(f'Run not found: {name}')




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
class Run(Configurable):
	'''
	Holds all the data and functions to load, save, train, and evaluate runs.
	Runs include the model, datasets, dataloaders, the logger, and stats
	'''
	def __init__(self, A, silent=False, **kwargs):
		super().__init__(A, **kwargs)
		self.silent = A.pull('silent', silent)

		self.invisible = A.pull('invisible', False)

		A = A.get_root()
		A = self._find_origin(A)
		
		if not self.invisible:
			self._setup_storage(A)

		self._prep(A)
		
		self.purge() # reset payload objs

		self.manual_prep()

	def manual_prep(self):
		pass
  
	def __repr__(self):
		return f'RUN:{self.get_name()}'
	
	def __str__(self):
		return self.get_name()
	
	def _prep(self, A):
		if 'clock' not in A:
			A.push('clock._type', 'clock', overwrite=False, silent=True)
		self.clock = A.pull('clock', ref=True)

		A.push('records._type', 'records', overwrite=False, silent=True)
		self.records = A.pull('records', ref=True)

		self.A = A
	
	def purge(self): # remove any potentially large objects to free memory
		self.dataset = None
		self.model = None
		
		self.loader = None
		self._loader = None
		
		self.batch = None
		self.out = None
		
		self.clock.clear()
		self.records.purge()
		
		self.checkpointer = None
		self.validation = None
		self.vizualizer = None
		
		self.results = {}
	
		self._started = False
	
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
				print(f'Loading Config (for run): {config_path}')
			
			load_A = fig.get_config(str(config_path))
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
		
		ckpt = str(ckpt)
		A.push('dataset._load-ckpt', ckpt, overwrite=True, silent=self.silent)
		A.push('model._load-ckpt', ckpt, overwrite=True, silent=self.silent)
		A.push('records._load-ckpt', ckpt, overwrite=True, silent=self.silent)
		A.push('clock._load-ckpt', ckpt, overwrite=True, silent=self.silent)
	
	# endregion

	def _gen_name(self, A):
		return 'test'
	
	@staticmethod
	def wrap_pull(A, keep_instance=True, **kwargs):
		A.begin()
		
		obj = A.pull_self()
		
		store_keys = A.pull('store_keys', [], silent=True)
		
		stored = {}
		
		if len(store_keys):
			
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
			
		if A.pull('keep_instance', keep_instance, silent=True):
			A.push('__obj', obj, silent=True)
		
		return obj
	
	def create_dataset(self, A=None, **meta):
		if A is None:
			A = self.get_config().sub('dataset')
		return self.wrap_pull(A, **meta)
	def create_model(self, A=None, **meta):
		if A is None:
			A = self.get_config().sub('model')
		return self.wrap_pull(A, **meta)
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
	
	def get_name(self):
		return self.name
	
	def get_description(self, ticks=None):
		if ticks is None:
			ticks = self.clock.get_time()
		
		tm = time.strftime("%H:%M:%S")
		
		progress = self.model.get_description()
		limit = self.clock.get_limit()
		
		mode = 'train'
		epochs = self.records['total_epochs'][mode]
		
		return f'[ {tm} ] {mode}:{epochs} {ticks}/{limit} {progress}'
	
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
	
	def get_loader(self, mode=None, **kwargs):
		return self.get_dataset().get_loader(mode, **kwargs)
	
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

	def get_batch(self):
		if self.batch is None:
			self.batch = self.get_dataset().get_batch()
		return self.batch
	
	def get_output(self):
		if self.out is None:
			self.out = self.get_model().step(self.get_batch())
		return self.out

	# endregion
	
	# region Training Phases
	
	def take_steps(self, n=1, complete=False):
		
		if not self._started:
			self.startup()
		
		if complete:
			n = self.clock.get_remaining()
	
		out = self._take_steps(n)
		
		if self.clock.get_remaining() <= 0:
			print('Training Complete')
			ticks = self.clock.get_time()
			if self.records is not None and not self.records.check(ticks, self):
				self.records.activate(ticks, info=self)
			if self.vizualizer is not None and not self.vizualizer.check(ticks, self):
				self.vizualizer.activate(ticks, info=self)
			if self.validation is not None and not self.validation.check(ticks, self):
				self.validation.activate(ticks, info=self)
			if self.checkpointer is not None and not self.checkpointer.check(ticks, self):
				self.checkpointer.activate(ticks, info=self)
		
		return out
	
	def _take_steps(self, n=1):
		for i in range(n):
			self._take_step()
			
	def _take_step(self):
		self.clock.tick(self)
	
	def receive_batch(self, batch):
		self.out = None
		self.batch = batch
	
	def receive_output(self, batch, out):
		self.out = out
		
		records = self.get_records()
		records['total_steps']['train'] += 1
		records['total_samples']['train'] += batch.size(0)
	
	def startup(self):
		
		self._started = True
		
		A = self.get_config()
		clock = self.get_clock()

		dataset = self.get_dataset()
		clock.register_alert('data', dataset)
		model = self.get_model()
		clock.register_alert('train', model)
		
		clock.sort_alerts(start_with=['data', 'train'])
		
		records = self.get_records()
		clock.register_alert('log', records)
		
		if 'viz' not in A:
			A.push('viz._type', 'run/viz', overwrite=False)
		self.vizualizer = A.pull('viz', None)
		if self.vizualizer is not None:
			clock.register_alert('viz', self.vizualizer)
		
		self.validation = A.pull('validation', None)
		if self.validation is None:
			print('No validation')
		else:
			clock.register_alert('val', self.validation)
		
		if self.invisible:
			print('No checkpointing')
		else:
			A.push('checkpoint._type', 'run/checkpoint', overwrite=False, silent=True)
			self.checkpointer = A.pull('checkpoint', None)
		
			if self.checkpointer is None:
				print('No checkpointer found')
			else:
				clock.register_alert('save', self.checkpointer)
		
		if 'print' not in A:
			A.push('print._type', 'run/print')
		print_step = A.pull('print', None)
		if print_step is not None:
			clock.register_alert('print', print_step)
		
		ckpt = A.pull('clock._load-ckpt', None, silent=self.silent)
		if ckpt is not None:
			ckpt = Path(ckpt)
			clock.load_checkpoint(ckpt)
		
		path = self.get_path()
		if path is not None:
			config_path = self.get_path() / 'config.yaml'
			if not config_path.is_file():
				A.export(config_path)
		
		mode = 'train'
		
		dataset.switch_to(mode)
		model.switch_to(mode)
		records.switch_to(mode)
		
		clock.prep(self)
		# dataset.prep(model=model, records=records)
		# model.prep(dataset=dataset, records=records)
		# records.prep(dataset=dataset, model=model)
		
		if mode not in records['total_steps']:
			records['total_steps'][mode] = 0
		if mode not in records['total_samples']:
			records['total_samples'][mode] = 0
		if mode not in records['total_epochs']:
			records['total_epochs'][mode] = 0
			
		# sync clock
		
		clock.set_time(records['total_steps'][mode])
		clock.sort_alerts(end_with=['save', 'print'], strict=False)
		
		if clock.get_remaining() is None:
			steps = A.pull('training.steps', '<>budget', None)
			if steps is None:
				epochs = A.pull('training.epochs', '<>epochs', None)
				if epochs is None:
					raise Exception('Clock has no limit')
				steps = len(dataset.get_loader()) * epochs
				
			clock.set_limit(steps)
		
		# self.loader = None
		# self.get_loader(activate=True, mode=mode, infinite=True)
	
	def prep_eval(self):
		raise NotImplementedError
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
		raise NotImplementedError
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
		
	def exit_run(self, cause, code=None, checkpoint=True):
		
		cmsg = f' (code={code})'
		
		if not self.silent:
			print(f'Exiting due to {cause}{cmsg}')
			
		if checkpoint and self.checkpointer is not None:
			print('Saving checkpoint')
			self.checkpointer.checkpoint()
		
		if code is not None:
			sys.exit(code)
	
	# endregion

@fig.AutoModifier('inline')
class Inline(Run):
	def _prep(self, A):
		
		inline = A.pull('inline', False)
		if inline:
			A.push('pbar._type', 'progress-bar', silent=True, overwrite=False)
		self.pbar = A.pull('pbar', None)
		if self.pbar is not None:
			A.push('print', None)

		super()._prep(A)

	def _take_steps(self, n=1):
		
		if self.pbar is not None and n > 1:
			self.pbar.init_pbar(limit=n)
		
		out = super()._take_steps(n=n)
		
		if self.pbar is not None:
			self.pbar.reset()
			
		return out
	
	def _take_step(self):
		
		super()._take_step()
		
		if self.pbar is not None:
			self.pbar.update(self.get_inline_description())
	
	def get_inline_description(self):
		
		progress = self.model.get_description()
		ticks = self.clock.get_time()
		limit = self.clock.get_limit()
		
		mode = 'train'
		epochs = self.records['total_epochs'][mode]
		
		return f'{mode}:{epochs} {ticks}/{limit} {progress}'
		
		
# @fig.AutoModifier('extendable')
# class Extendable(Run):
#
# 	def startup(self, A):
# 		super().startup()
#
# 		A = self.get_config()
#
# 		extend = A.pull('extend', None)
# 		if extend is not None:
# 			self.clock.set_time(extend)


		
@fig.AutoModifier('timed-run')
class Timed(Run):
	
	def startup(self):
		super().startup()
		
		A = self.get_config()

		time_limit = A.pull('time_limit', None)
		if time_limit is not None:
			if not self.silent:
				print(f'SOFT TIME LIMIT: {time_limit:2.2f} hrs')
			self.timer_start = time.time()
			time_limit = 3600 * time_limit
		
			self.timer_exit_code = A.pull('time_limit_exit_code', 3)
		
		self.timer_limit = time_limit

	def save_checkpoint(self, save_model=True):
		super().save_checkpoint(save_model=save_model)
		
		time_limit = self.timer_limit
		if time_limit is not None:
			start_time = self.timer_start
			if (time.time() - start_time) > time_limit:
				self.exit_run('timelimit reached', code=self.timer_exit_code)
			

		