import sys, os
from pathlib import Path
import time
import socket
import random

from collections import OrderedDict

import humpack as hp

from omnibelt import load_yaml, save_yaml, get_now, create_dir, get_printer, unspecified_argument

import omnifig as fig
from omnifig import Configurable
from omnifig.errors import MissingParameterError, MissingComponentError

prt = get_printer(__name__)


_warn_savedir = True
def get_save_dir(A=None, silent=False):
	if A is not None:
		root = A.pull('saveroot', None, silent=silent)
		if root is not None:
			return Path(root)
	
	if 'OMNILEARN_SAVE_DIR' in os.environ:
		return Path(os.environ['OMNILEARN_SAVE_DIR'])
	
	root = Path(os.getcwd())
	root = root / 'trained_nets'
	root.mkdir(exist_ok=True)
	
	global _warn_savedir
	if _warn_savedir:
		prt.warning(f'No savedir found (specify with "OMNILEARN_SAVE_DIR" env variable), '
		            f'now using {str(root)}')
		_warn_savedir = False
	
	return root



class RunNotFoundError(Exception):
	def __init__(self, name):
		super().__init__(f'Run not found: {name}')



# class Validation_Flag(Exception):
# 	pass
#
# class Checkpoint_Flag(Exception):
# 	pass
#
# class NoOverwriteError(Exception):
# 	pass


@fig.Script('load-run', description='Load a new or existing run') # the script just loads a run
def load_run(A):
	
	silent = A.pull('silent', False, silent=True)
	
	override = A.pull('override', None, raw=True, silent=True)
	
	name = A.pull('path', '<>load', '<>resume', None, silent=silent)
	if name is not None:
		base = Path(name)
		path = base
		
		if not path.is_dir():
			saveroot = get_save_dir(A)
			path = saveroot / base
		
		if not path.is_dir():
			root = A.pull('root', None, silent=silent)
			if root is not None:
				path = root / base
		
		if not path.is_dir():
			raise RunNotFoundError(name)
		
		A = fig.get_config(str(path))
		if override is not None:
			A.update(override)
		A.push('path', str(path))
	
	if 'run' in A:
		config = A.pull('run', None, raw=True)
		if config is not None:
			A = config

	run_type = A.push('_type', 'run', overwrite=False, silent=True)
	if not fig.has_component(run_type):
		prt.error(f'Run type "{run_type}" not found, using default instead')
		A.push('_type', 'run')
		
	return A.pull_self()
	
	
@fig.Component('run')
class Run(Configurable):
	'''
	Holds all the data and functions to load, save, train, and evaluate runs.
	Runs include the model, datasets, dataloaders, the logger, and stats
	'''
	def __init__(self, A, name=unspecified_argument, path=unspecified_argument,
	             silent=None, invisible=None, use_root=None, **kwargs):
		
		if silent is None:
			silent = A.pull('silent', silent)

		if invisible is None:
			invisible = A.pull('invisible', False, silent=silent)

		if use_root is None:
			use_root = A.pull('use_config_root', True, silent=silent)
			
		if path is unspecified_argument:
			path = A.pull('path', '<>load', '<>resume', None, silent=silent)
		
		if name is unspecified_argument:
			name = A.pull('name', None, silent=silent) if path is None else None
		
		super().__init__(A, **kwargs)
		self.silent = silent
		self.invisible = invisible
		if use_root:
			A = A.get_root()
		self.A = A
		
		self.name = name
		
		if path is not None:
			A.push('path', path, silent=silent)
			path = Path(path)
		self.path = path
		self.novel = path is None
		
		if invisible:
			raise NotImplementedError
		# if not invisible:
		# 	self._setup_storage(A)
		self._setup_storage(A)

		self._prep(A)
		
		self.purge() # reset payload objs

		self.manual_prep()

	def manual_prep(self):
		pass
  
	def __repr__(self):
		return f'{self.__class__.__name__.upper()}:{self.get_name()}'
	
	def __str__(self):
		return self.get_name()
	
	def _prep(self, A):
		if 'clock' not in A:
			A.push('clock._type', 'clock', overwrite=False, silent=True)
		self.clock = A.pull('clock', ref=True)

		A.push('records._type', 'records', overwrite=False, silent=True)
		# self.records = A.pull('records', ref=True)
		self.records = None
		self.get_records()


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
		
	def _setup_storage(self, A):
		if self.path is None:
			self._create_storage(A)
		else:
			self._load_storage(A)
		
	def _create_storage(self, A):
		
		name = self.get_name()
		
		if name is None:
			name = self._gen_name(A)
		
		logdate = A.pull('name-include-date', True)
		if logdate:
			now = get_now()
			name = f'{name}_{now}'
		
		self.name = name
		
		path = self.get_path()
		
		if path is None and not self.invisible:
			saveroot = get_save_dir(A, silent=self.silent)

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
			A = self.get_config().pull('dataset', raw=True, silent=True)
		return self.wrap_pull(A, **meta)
	def create_model(self, A=None, **meta):
		if A is None:
			A = self.get_config().pull('model', raw=True, silent=True)
		return self.wrap_pull(A, **meta)
	def create_records(self, A=None, **meta):
		if A is None:
			A = self.get_config().pull('records', raw=True, silent=True)
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
	
	def get_results(self, ident, path=None, remove_ext=True, **kwargs): # you must specify which results (ident used when results were created)
		
		if ident in self.results:
			return self.results[ident]

		fixed = ident.split('.')[0] if remove_ext else ident

		if fixed in self.results:
			return self.results[fixed]

		self.results[fixed] = self._load_results(name=ident, path=path, **kwargs)

		return self.results[fixed]


	def update_results(self, ident, data, path=None, overwrite=False, remove_ext=True):

		fixed = ident.split('.')[0] if remove_ext else ident

		if self.has_results(fixed, path=path) and not overwrite:
			old = self.get_results(fixed, path=path)
			old.update(data)
			data = old

		self._save_results(data, name=fixed, path=path)
		
		if fixed in self.results:
			self.results[fixed] = data
			
		return data


	def has_results(self, ident, path=None, ext=None):
		fixed = ident.split('.')[0] if ext is not None else ident
		
		if fixed in self.results:
			return True
		
		if ext is not None:
			fixed = f'{fixed}.{ext}'
		
		if path is None:
			path = self.get_path()
		return (path/fixed).is_file()

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
		
		print(records['total_steps'][mode])
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
	
	
	def evaluate(self, ident='eval', config=None,
	             evaluation=None, store_batch=True, overwrite=False, path=None):
		
		if config is not None:
			self.get_config().sub('eval').update(config)
		config = self.get_config().sub('eval')
		
		ident = config.push('ident', ident, overwrite=False)
		mode = config.pull('mode', 'val')
		assert ident is not None, 'No ident specified'
		
		config.push('mode', mode, force_root=True, silent=True)
		
		if path is None:
			path = self.get_path()
		if path is not None:
			path = Path(path)
			print(f'Will save results to {str(path)}')
			
		dataset = self.get_dataset()
		model = self.get_model()
		records = self.get_records()
		
		dataset.switch_to(mode)
		model.switch_to(mode)
		records.switch_to(mode)
		
		N = self.get_clock().get_time()
		fmt = '{}/{}'.format(ident, '{}')
		if config is not None:
			fmt = config.pull('log-fmt', fmt)
			
		_records_info = records.get_fmt(), records.get_step()
		records.set_fmt(fmt)
		records.set_step(N)
		
		if 'evaluations' not in records:
			records['evaluations'] = {}
		if ident not in records['evaluations']:
			records['evaluations'][ident] = []
		if N in records['evaluations'][ident]:
			overwrite = overwrite if config is None else config.pull('overwrite', False)
			if not overwrite:
				print(f'Already evaluated "{ident}" after {N} steps, and overwrite is False')
				results = None
				if config is not None and config.pull('load-results', False):
					results = self.get_results(ident)
				return results
			print(f'Overwriting {ident} at step {N}')
		else:
			records['evaluations'][ident].append(N)
		
		store_batch = store_batch if config is None else config.pull('store_batch', store_batch)
		
		if evaluation is None and config is not None:
			evaluation = config.pull('evaluation', None)
		
		batch, output = None, None
		if evaluation is not None:
			evaluation.set_mode(mode)
			def step(_batch, step, _records=None):
				nonlocal output, batch
				out = evaluation.epoch_step(_batch, step)
				if output is None:
					try:
						model.visualize(out, records)
					except AttributeError:
						pass
					batch = _batch
					output = out
				return out
			
			evaluation.run_epoch(dataset=dataset, model=model, records=None, step_fn=step)
		
			self.batch = batch
			self.output = output
		
		try:
			eval_out = model.evaluate(self, config=config)
		except AttributeError:
			pass
		else:
			if output is None:
				output = eval_out
			elif eval_out is not None:
				output.update(eval_out)
		
		records.step(fmt=fmt)
		
		if store_batch and output is not None:
			output['batch'] = batch
		if path is not None:
			self.update_results(ident, {k:v for k,v in output.items()}, path=path)
			records.checkpoint(path)
		
		fmt, N = _records_info
		records.set_fmt(fmt)
		records.set_step(N)
		
		return output
		
		
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


@fig.AutoModifier('testable')
class Testable(Run):

	def purge(self):
		super().purge()
		self.trainset = None
		self.testset = None

	def create_dataset(self, mode, A=None, switch=False, **meta):
		if A is None:
			A = self.get_config().pull(f'{mode}-dataset', None, raw=True, silent=True)
			if A is None:
				A = self.get_config().pull('dataset', raw=True, silent=True)
				A.begin()
				A.push('mode', mode, silent=True)
				switch = True
		dataset = self.wrap_pull(A, **meta)
		if switch:
			dataset.switch_to(mode)
		return dataset
	def create_trainset(self, A=None, **meta):
		return self.create_dataset(mode='train', A=A, **meta)
	def create_testset(self, A=None, **meta):
		return self.create_dataset(mode='test', A=A, **meta)

	def get_dataset(self, mode='train', **meta):
		if mode == 'test':
			return self.get_testset(**meta)
		return self.get_trainset(**meta)
	def get_trainset(self, **meta):
		if self.trainset is None:
			self.trainset = self.create_trainset(**meta)
		return self.trainset
	def get_testset(self, **meta):
		if self.testset is None:
			self.testset = self.create_testset(**meta)
		return self.testset

	def get_loader(self, mode='train', **kwargs):
		if mode == 'test':
			return self.get_testloader(**kwargs)
		return self.get_trainloader(**kwargs)
	def get_trainloader(self, mode='train', **kwargs):
		return self.get_trainset().get_loader(mode, **kwargs)
	def get_testloader(self, mode='test', **kwargs):
		return self.get_trainset().get_loader(mode, **kwargs)


@fig.AutoModifier('inline')
class Inline(Run):
	def _prep(self, A):

		super()._prep(A)
		
		inline = A.pull('inline', False)
		if inline:
			A.push('pbar._type', 'progress-bar', silent=True, overwrite=False)
		self.pbar = A.pull('pbar', None)
		if self.pbar is not None:
			A.push('print', None)


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
			self.pbar.update(desc=self.get_inline_description())
	
	def get_inline_description(self):
		
		progress = self.model.get_description()
		ticks = self.clock.get_time()
		limit = self.clock.get_limit()
		
		mode = 'train'
		epochs = self.records['total_epochs'][mode]
		
		return f'{mode}:{epochs} {ticks}/{limit} {progress}'

		
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
			

		