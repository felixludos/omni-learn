
import sys, os
import time
import socket
import random

import humpack as hp

import omnifig as fig

from omnibelt import load_yaml, save_yaml, get_now, create_dir
# from .. import util
# from .paths import


FD_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(FD_PATH), 'trained_nets')


# def get_saveroot(A=None, silent=False, additional_keys=()):
# 	root = None
# 	if A is not None:
# 		root = A.pull('saveroot', '<>save_path', *additional_keys, None, silent=silent)
#
# 	if root is None:
# 		root = os.environ['FOUNDATION_SAVE_DIR'] if 'FOUNDATION_SAVE_DIR' in os.environ else DEFAULT_SAVE_PATH
#
# 	return root


class NoConfigFoundError(Exception):
	def __init__(self, msg):
		super().__init__(msg)


def wrap_script(script_name, A, **kwargs):
	A.begin()
	
	obj = fig.run(script_name, A, **kwargs)
	
	store_keys = A.pull('store_keys', [])
	
	stored = {}
	for key in store_keys:
		try:
			val = A.pull(key, silent=True)
		except fig.MissingConfigError:
			pass
		else:
			stored[key] = val
	
	A.abort()
	
	for key, val in stored.items():
		A.push(key, val, silent=True, force_root=True)
	
	return obj


def find_config(ident, check_root=True):
	path = ident
	if os.path.isfile(ident):
		if 'config.yaml' in ident:
			return ident
		path = os.path.dirname(ident)
	
	if os.path.isdir(path):
		for fname in os.listdir(path):
			if fname == 'config.yaml':
				return os.path.join(path, fname)
	
	root = os.environ['FOUNDATION_SAVE_DIR'] if 'FOUNDATION_SAVE_DIR' in os.environ else DEFAULT_SAVE_PATH
	
	if check_root:
		return find_config(os.path.join(root, ident), check_root=False)
	raise NoConfigFoundError(f'no config found: {ident}')


def _get_ckpt_num(ident):
	ident = os.path.basename(ident)
	
	val = ident.split('.')[0].split('_')[-1]
	try:
		return int(val)
	except ValueError:
		pass
	
	return None


def _find_ckpt_with(ident, last=False, req='model', check_root=True):
	# valid = lambda n: ((last and 'ckpt' in n and 'best' not in n)
	#                    or (not last and 'best' in n)) and req in n
	valid = lambda n: f'ckpt-{req}' in n
	
	path = ident
	
	if os.path.isfile(ident):
		if valid(ident):
			return ident
		path = os.path.dirname(ident)
	
	if os.path.isdir(path):
		names = [fname for fname in os.listdir(path) if valid(fname)]
		
		if len(names) == 1:
			return os.path.join(path, names[0])
		elif len(names) > 1:
			nums = [_get_ckpt_num(fname) for fname in names]
			
			name = max(zip(names, nums), key=lambda x: x[1] if x[1] is not None else -1)[0]
			return os.path.join(path, name)
	
	root = os.environ['FOUNDATION_SAVE_DIR'] if 'FOUNDATION_SAVE_DIR' in os.environ else DEFAULT_SAVE_PATH
	# return _find_ckpt_with(os.path.join(root, ident), last=last, req=req)
	
	if check_root:
		return _find_ckpt_with(os.path.join(root, ident), last=last, req=req, check_root=False)
	raise NoConfigFoundError(f'no checkpoint found: {ident}')


def find_checkpoint(ident, last=False):
	return _find_ckpt_with(ident, last=last, req='model')


def find_records(ident, last=False):
	return _find_ckpt_with(ident, last=last, req='records')


class Validation_Flag(Exception):
	pass

class Checkpoint_Flag(Exception):
	pass

class NoOverwriteError(Exception):
	pass

@fig.Script('load_run', description='Load a new or existing run') # the script just loads a run
@fig.Component('run')
class Run:
	'''
	Holds all the data and functions to load, save, train, and evaluate runs.
	Runs include the model, datasets, dataloaders, the logger, and stats
	'''
	def __init__(self, A, silent=False):
		
		self.records = A.pull('records', {})
		self.silent = A.pull('silent', silent)
		self.rng = random.Random()
		
		skip_load = A.pull('skip_run_load', False)
		
		self.A = A.get_root()
		if not skip_load:
			self.A = self._find_origin(self.A) # note that this uses the root
			
			self.name = self._setup_storage(self.A)
		elif not self.silent:
			print('WARNING: did not check run origin')
		
		self.purge() # reset payload objs
	
	def __repr__(self):
		return 'Run({})'.format(getattr(self, 'save_dir', ''))
	
	def __str__(self):
		return 'Run({})'.format(getattr(self, 'save_dir', ''))
	
	def _increment_rng(self, seed):
		self.rng.seed(seed)
		return self.rng.getrandbits(32)
	
	def purge(self): # remove any potentially large objects to free memory
		self.model = None
		self.datasets = {}
		self.loaders = {}
		self.logger = None
		self.stats = None
		self.viz_criterion = None
		
		self.active_stage = None
		self.train_state = hp.adict()
		
		self.results = {}
	
	def _find_origin(self, A):
		
		last = A.pull('last', False)
		
		path = A.pull('load', None)
		novel = True
		if path is None:
			path = A.pull('resume', '<>path', None)
			if path is not None:
				novel = False
		
		raw_path = path
		if path is not None:
			path = find_config(path)
		
		novel = A.push('novel', novel, overwrite=False)
		override = A.pull('override', {})
		extend = A.pull('extend', None)
		
		if path is not None:
			if not self.silent:
				print(f'Loading Config: {raw_path}')
			
			load_A = fig.get_config(path)
			if novel:
				load_A.update(A)
			A.clear()
			A.update(load_A)
			A.update(override)
		
		if extend is not None:
			A.push('training.step_limit', extend)
		
		if novel:
			self.records.update(self._initialize_records(A))
		
		if path is not None:
			self._load_records(path, last=last)
			self._set_model_ckpt(path, last=last)
		
		return A
		
	def _initialize_records(self, A):
		
		records = {
			'total_steps': 0,
			
			'total_samples': {'train': 0, 'val': 0},
			'total_epochs': {'train': 0, 'val': 0},
			'stats': {'train': [], 'val': []},
			
			'batch': 0,
			'checkpoint': None,
			'val': 0,
			
			'epoch_seed': self._increment_rng(A.pull('seed', self.rng.getrandbits(32))),
			
			'history': A.pull('_history', [], silent=False),
			'argv': sys.argv,
			'timestamp': get_now(),
		}
		
		track_best = A.pull('training.track_best', False)
		
		if track_best:
			records['best'] = {'loss': None, 'checkpoint': None}
		
		# tau = info.pull('stats_decay', 0.001)
		# util.set_default_tau(tau)
		
		return records
		
	def _setup_storage(self, A):
		
		name = A.pull('run.name', '<>name', None)
		if name is None:
			name = self._gen_name(A)
			A.push('name', name)
		
		save_dir = None
		save_path = None
		
		A.push('training.stats._type', 'stats', overwrite=False)
		
		invisible = A.pull('invisible', False)
		if not invisible:

			save_dir = name
			logdate = A.pull('output.logdate', True)
			if logdate:
				now = self.get_timestamp()
				save_dir = f'{name}_{now}'
			
			save_dir = A.push('output.save_dir', save_dir)
		
			saveroot = A.pull('output.saveroot',
			                  os.environ['FOUNDATION_SAVE_DIR']
			                  if 'FOUNDATION_SAVE_DIR' in os.environ
			                  else DEFAULT_SAVE_PATH)
			
			save_path = os.path.join(saveroot, save_dir)
			
			if not os.path.isdir(save_path):
				create_dir(save_path)
			
		self.save_dir = save_dir
		self.save_path = save_path
		
		return name
		
	def _load_records(self, path=None, last=False):
		
		try:
			records_path = find_records(path, last=last)
		except NoConfigFoundError:
			pass
		else:
			records = load_yaml(records_path)
			
			self.records.update(records)

	def _set_model_ckpt(self, path=None, last=False): # don't auto load yet (just prepare for loading)
		
		if path is not None:
			ckpt_path = find_checkpoint(path, last=last)
			self.A.push('model._load_params', ckpt_path, overwrite=True, silent=self.silent)
		
	def _set_dataset_ckpt(self, path=None, last=False):
		pass
		
	def __getattr__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError as e:
			if item in self.records:
				return self.records[item]
			raise e

	def _gen_name(self, A):
		return None
	
	def _get_path_from_ident(self, ident, ext='pth.tar'):
		if self.save_path is None:
			return None
		return os.path.join(self.save_path, f'{ident}.{ext}')
	
	def create_logger(self, save_path=None):
		
		A = self.get_config()
		
		if save_path is None:
			save_path = self.save_path
		elif not self.silent:
			print(f'WARNING: using a foreign path: {save_path}')
		
		if save_path is None:
			return None
		
		# tblog = A.pull('output.tblog', False)
		# txtlog = A.pull('output.txtlog', False)
		#
		# if tblog or txtlog:
		# 	logtypes = []
		# 	if txtlog:
		# 		logtypes.append('stdout')
		# 	if tblog:
		# 		logtypes.append('on tensorboard')
		#
		# 	if not self.silent:
		# 		print('Logging {}'.format(' and '.join(logtypes)))
		
		A.begin() # TODO: clean up
		A.push('output.logger._type', 'logger', overwrite=False)
		A.push('output.logger.log_dir', save_path, overwrite=True)
		
		logger = A.pull('output.logger')
		A.abort()
		return logger
		
	def create_stats(self, *names, model_stats_fmt=None, include_loss=True, silent=False):
		
		A = self.get_config()
		
		stats = A.pull('training.stats', silent=silent)
		
		if include_loss:
			stats.new('loss')
		
		for name in names:
			stats.new(name)
		
		if model_stats_fmt is not None:
			model = self.get_model()
			try:
				stats.shallow_join(model.stats, fmt=model_stats_fmt)
			except AttributeError:
				pass
		
		return stats
	
	def create_model(self, **meta):
		return wrap_script('load_model', self.A.sub('model'), **meta)
	
	def create_datasets(self, *names, **meta):
		
		A = self.get_config().sub('dataset')
		
		keep_going = True
		name = None
		datasets = {}
		while keep_going:
			for name in names:
				if name not in datasets:
					A.begin()
					A.push('mode', name, overwrite=True)
					break

			result = wrap_script('load_data', A, **meta)

			if isinstance(result, dict) and name in result:
				datasets.update(result)
			else:
				datasets[name] = result

			if name is not None and name not in datasets:
				raise Exception(f'Failed to create dataset: {name}')
			
			keep_going = False
			for name in names:
				if name not in datasets:
					keep_going = True
			
		return datasets
	
	# region "Payload" objects - not loaded automatically

	def get_config(self):
		return self.A

	def get_model(self, **meta):
		if self.model is None:
			self.model = self.create_model(**meta)
		return self.model
	def get_datasets(self, *names, **meta):
		missing = [name for name in names if name not in self.datasets]
		if len(self.datasets) == 0 or len(missing):
			self.datasets.update(self.create_datasets(*missing, **meta))
		if len(names) == 1:
			return self.datasets[names[0]]
		if len(names) >= 1:
			return {n:d for n,d in self.datasets if n in names}
		return self.datasets.copy()
	def get_logger(self, path=None):
		if path is not None:
			return self._create_logger(self.A, save_path=path)
		if self.logger is None:
			self.logger = self.create_logger(save_path=path)
		return self.logger
	
	def get_loaders(self, *dataset_names, **datasets):
		A = self.get_config().sub('dataset')
		single_dataset = len(dataset_names) == 1
		
		if len(datasets) == 0:
			datasets = self.get_datasets(*dataset_names)
			if single_dataset:
				datasets = {dataset_names[0] : datasets}
		
		for name, dataset in datasets.items():
			if name not in self.loaders:
				self.loaders[name] = dataset.to_loader(A)
		
		if single_dataset:
			return self.loaders[next(iter(datasets.keys()))]
		return {name:self.loaders[name] for name in datasets}
		
	def get_results(self, ident): # you must specify which results (ident used when results were created)
		
		if ident in self.results:
			return self.results[ident]
		
		raise NotImplementedError # TODO: find results with ident, load, and return

	def get_training_datasets(self):
		return {name:dataset for name,dataset in self.get_datasets().items() if name != 'test'}

	def get_stats(self, *args, purge_old=False, **kwargs):
		if self.stats is None or purge_old:
			self.stats = self.create_stats(*args, **kwargs)
		return self.stats
	
	
	# endregion
	
	# region Signals

	def get_total_steps(self):
		return self.records['total_steps']

	def get_total_samples(self, mode='train'):
		return self.records['total_samples'][mode]

	def get_timestamp(self):
		return self.records['timestamp']

	def get_save_path(self):
		return self.save_path
	
	def get_save_dir(self):
		return self.save_dir

	def keep_going(self):
		sample_limit = self.train_state.sample_limit
		step_limit = self.train_state.step_limit
		return (sample_limit is None
		        or self.get_total_samples('train') < sample_limit) \
			and self.records['total_steps'] < step_limit

	def print_time(self):
		print_freq = self.train_state.print_freq
		return print_freq is not None and self.get_total_steps() % print_freq == 0

	def log_time(self):
		logger = self.get_logger()
		log_freq = self.train_state.log_freq
		return logger is not None and log_freq is not None and self.get_total_steps() % log_freq == 0
	
	def val_time(self):
		val_freq = self.train_state.val_freq
		return val_freq is not None and (self.get_total_steps() % val_freq == 0 or not self.keep_going())
	
	def save_time(self):
		save_freq = self.train_state.save_freq
		return save_freq is not None and ((save_freq > 0 and self.get_total_steps() % save_freq == 0)
		                                                    or not self.keep_going())
	
	# endregion
	
	# region Saving
	
	def save_checkpoint(self, root=None, save_model=True):  # TODO: add an option to save dataset/loader
		Q = self.train_state
		start = time.time()

		if root is None:
			root = self.save_path
		
		steps = self.get_total_steps()

		if root is not None and os.path.isdir(root):
			
			records = self.records
			records['checkpoint'] = steps
			is_best = self.train_state.is_best
			
			model = self.get_model()
			
			model_paths = []
			
			if steps is not None:
				if records is not None:
					rpath = os.path.join(root, f'ckpt-records_{steps}.yaml')
					save_yaml(records, rpath)

				mpath = os.path.join(root, f'ckpt-model_{steps}.pth.tar')
				model_paths.append(mpath)
			
			if is_best:
				if records is not None:
					rpath = os.path.join(root, f'ckpt-records_best.yaml')
					save_yaml(records, rpath)
				
				mpath = os.path.join(root, f'ckpt-model_best.pth.tar')
				model_paths.append(mpath)
			
			if save_model:
				model.save_checkpoint(*model_paths)
			
			best_info = ' (new best) ' if Q.is_best else ''
			Q.is_best = False
			
			if not self.silent:
				print(f'[[ Saved checkpoint {steps}{best_info} to {root} ]]')
			
			Q.time_stats.update('save', time.time() - start)
	
		elif not self.silent:
			print(f'[[ no checkpoint {steps} saved ]]')
	
	def save_results(self, name, results, root=None):
		
		if root is None:
			root = self.save_path
			
		if root is not None and os.path.isdir(root):
			
			path = self._get_path_from_ident(name, 'pth.tar')
			
			self.get_model().save_data(path, data=results)
			
			print(f'[[ {name} results saved to {path} ]]')
		
		else:
			print(f'[[ no results saved, as no root was provided ]]')
	
	# endregion
	
	# region Training Phases
	
	def prepare(self):
		'''
		Loads model and datasets, and preps them (linking them to each other, if necessary)
		
		:return:
		'''
		
		model = self.get_model()
		trainsets = self.get_training_datasets()
		
		model.prep(trainsets)
		for dataset in trainsets.values():
			dataset.prep(model)
	
	def continuous(self, pbar=None):
		
		self.startup()
		stats = self.train_state.train_stats
		
		self.new_epoch()
		
		restart_pbar = lambda: None if pbar is None else pbar(total=self.train_state.step_limit,
		                                     initial=self.get_total_steps())
		
		
		bar = restart_pbar()
		
		records = self.records
		step_limit = self.train_state.step_limit
		mode = 'train'
		
		while self.keep_going():
		
			out = self.train_step(force_step=True)
		
			if self.log_time():
				self.log_step(out, '{}/train', measure_time=False)

				sys.stdout.flush()
		
			if self.val_time():
				if bar is not None:
					bar.close()
					
				self.validate('val', pbar=pbar)
				
				if self.keep_going():
					bar = restart_pbar()

				sys.stdout.flush()
		
			if self.save_time():
				if bar is not None:
					bar.close()
					
				self.save_checkpoint()
				
				if self.keep_going():
					bar = restart_pbar()
		
			if bar is not None:
				bar.update(1)
				title = '{} ({})'.format(mode, records['total_epochs'][mode] + 1) \
					if mode in records['total_epochs'] else mode
				loss_info = 'Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(),
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


		# self.exit_run('training complete')
	
	
	def startup(self):
		
		if self.save_path is not None:
			config_path = os.path.join(self.save_path, 'config.yaml')
			if not os.path.isfile(config_path):
				self.get_config().export(config_path)
		
		A = self.A
		Q = self.train_state
		
		Q.step_limit = A.pull('training.step_limit', None)
		Q.sample_limit = A.pull('training.sample_limit', None)
		assert Q.step_limit is not None or Q.sample_limit is not None, 'No limit provided'
		Q.val_freq = A.pull('training.val_freq', None)
		Q.skip_prev_batches = A.pull('training.skip_prev_batches', False)

		self.viz_criterion = A.pull('training.viz-criterion', None)
		
		model_stats_fmt = A.pull('training.model_train_stats_fmt', '<>training.model_stats_fmt', '{}')
		Q.train_stats = self.create_stats(model_stats_fmt=model_stats_fmt)
		Q.time_stats = A.pull('training.time_stats', '<>training.stats')
		Q.time_stats.new('data', 'model', 'viz', 'val', 'save')
		
		
		Q.save_freq = A.pull('output.save_freq', -1)
		Q.print_freq = A.pull('output.print_freq', None)
		Q.log_freq = A.pull('output.log_freq', None)
		Q.unique_tests = A.pull('output.unique_tests', False)
		Q.model_val_stats_fmt = A.pull('training.model_val_stats_fmt', '<>training.model_stats_fmt', '{}')
		Q.display_samples = A.pull('output.display_samples', False)  # count in terms of samples instead of iterations
		Q.display_smoothed = A.pull('output.display_smoothed', True)
		time_stats_fmt = A.pull('output.time_stats_fmt', 'time-{}')
		if time_stats_fmt is not None:
			Q.train_stats.shallow_join(Q.time_stats, fmt=time_stats_fmt)
		
		trainloader = self.get_loaders('train')
		
		valloader = self.get_loaders('val')
		if valloader is None:
			Q.val_freq = None
		
		save_freq = Q.get('save_freq', -1)
		if save_freq is not None and 0 < save_freq < len(trainloader):
			if not self.silent:
				print('WARNING: saving more than once per epoch: checkpoint every {} iterations'.format(save_freq))
			
			# assert save_freq > 100, 'not allowed to save more often than once every 100 steps -- remember 55-8'
			
			quick_save = A.pull('output.quick_save', False)  # force saving so frequently

			if not quick_save:
				raise Exception('not allowed to save more often than once every 100 steps')

			# if not quick_save:
			# 	save_freq = len(trainloader)
		if not self.silent and (save_freq is not None and save_freq > 0):
			print(f'Will save a checkpoint every {save_freq} steps')
		Q.save_freq = save_freq
	
		Q.is_best = False
	
	def start_loader(self, mode='train', dataloader=None):
		Q = self.train_state
		
		if dataloader is None:
			dataloader = self.get_loaders(mode)
		
		if mode == 'train':
			epoch_seed = self.records.get('epoch_seed', None)
			if epoch_seed is not None:
				dataloader.set_seed(epoch_seed)
				self.records['epoch_seed'] = self._increment_rng(epoch_seed)
		
		loader = iter(dataloader)
		if mode == 'train' and Q.skip_prev_batches:
			skip_batches = min(self.records.get('batch', 0), len(dataloader))
			try:
				loader.skip(skip_batches)
			except AttributeError:
				print('WARNING: no auto skip implemented')
				for _ in range(skip_batches):
					next(loader)
		
		assert len(dataloader) >= 1, 'no batches'
		
		Q.loader = loader
		
		return loader
	
	def new_epoch(self, mode='train'):
		
		model = self.get_model()
		model.pre_epoch(mode, self.records)
		self.get_datasets(mode).pre_epoch(mode, self.records)
		
		if mode == 'train':
			model.train()
		else:
			model.eval()
		
		return self.start_loader(mode)
		
	def end_epoch(self, mode='train', stats=None):
		
		self.get_model().post_epoch(mode, self.records, stats=stats)
		self.get_datasets(mode).post_epoch(mode, self.records, stats=stats)

		self.records['total_epochs'][mode] += 1
		self.records['stats'][mode].append(stats.export())
	
	def log_step(self, out, tag_fmt='{}/train', measure_time=True):
		Q = self.train_state
		logger = self.get_logger()
		train_stats = Q.train_stats
		
		start = time.time()
		
		if self.viz_criterion is not None:
			train_stats.update('loss-viz', self.viz_criterion(out).detach())
		
		logger.set_step(self.records['total_samples']['train'] if Q.display_samples else self.records['total_steps'])
		logger.set_tag_format(tag_fmt)
		
		display = train_stats.smooths() if Q.display_smoothed else train_stats.avgs()
		for k, v in display.items():
			logger.add('scalar', k, v)
		
		try:
			self.get_model().visualize(out, logger)
		except AttributeError:
			pass

		if measure_time:
			Q.time_stats.update('viz', time.time() - start)
		
	def train_step(self, force_step=False):
		
		Q = self.train_state
		time_stats = Q.time_stats
		
		if 'loader' not in Q or Q.loader is None:
			Q.loader = self.new_epoch('train')
		
		loader = Q.loader
		
		try:
			start = time.time()
			# batch = loader.next_batch()
			batch = next(loader)
			
		except StopIteration:
			self.end_epoch('train', Q.train_stats)
			
			if not force_step:
				raise StopIteration
		
			loader = self.new_epoch('train')
			Q.loader = loader

			self.records['batch'] = 0
			
			start = time.time()
			batch = next(loader)
		
		time_stats.update('data', time.time() - start)
		start = time.time()
		
		out = self.get_model().step(batch)
		
		if 'loss' in out:
			Q.train_stats.update('loss', out['loss'].detach())
		
		time_stats.update('model', time.time() - start)
		
		B = batch.size(0)
		self.records['total_samples']['train'] += B
		self.records['batch'] += 1
		self.records['total_steps'] += 1
	
		if len(loader) == 0:
			self.end_epoch('train', Q.train_stats)
			self.records['batch'] = 0
			Q.loader = None
	
		return out
	
	def validate(self, mode='val', pbar=None):
		if mode != 'val':
			raise NotImplementedError
		Q = self.train_state
		records = self.records
		model = self.get_model()
		
		train_model_stats = getattr(model, 'stats', None)
		stats = self.create_stats(model_stats_fmt=None, silent=True)
		
		if stats is not None and train_model_stats is not None and Q.model_val_stats_fmt is not None:
			
			model.stats = model.stats.copy()
			model.stats.reset()
			
			stats.shallow_join(model.stats, fmt=Q.model_val_stats_fmt)
		
		loader = self.new_epoch(mode)
		
		bar = None
		if pbar is not None:
			bar = pbar(total=len(loader))
		
		out = None
		
		title = '{} ({})'.format(mode, records['total_epochs'][mode]+1) \
			if mode in records['total_epochs'] else mode
		
		start = time.time()
		
		for batch in loader:
			
			B = batch.size(0)
			if mode in records['total_samples']:
				records['total_samples'][mode] += B
			
			out = model.test(batch)
			if 'loss' in out:
				stats.update('loss', out.loss.detach())
			
			if bar is not None:
				bar.update(1)
				loss_info = ' Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(),
				                                            stats['loss'].smooth.item()) \
					if stats['loss'].count > 0 else ''
				bar.set_description('{} ckpt={}{}'.format(title, records['checkpoint'], loss_info))

		if mode in Q.time_stats:
			Q.time_stats.update(mode, time.time() - start)
		
		if out is not None:
			self.log_step(out, '{}/{}'.format('{}', mode), measure_time=False)
		
		if bar is not None:
			bar.close()
		
		loss_info = ''
		if mode == 'val':
			val_loss = stats['loss']
			
			best_info = ''
			if 'best' in records and \
					records['best']['loss'] is None or (val_loss.count > 0
					                                    and val_loss.avg <= records['best']['loss']):
				prev = '!' if records['best']['loss'] is None \
					else ', previous was: {:.3f} (ckpt={})'.format(records['best']['loss'],
					                                               records['best']['checkpoint'])
				best_info = f' which is a new best{prev}'
				records['best']['loss'] = val_loss.avg.item()
				records['best']['checkpoint'] = records['checkpoint']
				Q.is_best = True
			loss_info = ' Loss: {:.3f}{}'.format(val_loss.avg.item(), best_info) \
				if val_loss.count > 0 else ''
		
		total_steps = self.get_total_steps()
		
		if not self.silent:
			print('[ {} ] {} Last={} Now={}/{}\n\t{}'.format(
				time.strftime("%H:%M:%S"), mode, records['checkpoint'],
				total_steps, Q.step_limit, loss_info))
			
		self.end_epoch(mode, stats)
		self.records[mode] = total_steps
		
		if train_model_stats is not None:
			model.stats = train_model_stats
	
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

		model = self.get_model()

		model.eval()

		results = model.evaluate(dataloader, logger=self.get_logger(),
		                         A=self.get_config().sub('eval'), run=self)

		if results is not None:
			ident = self.eval_identifier
			self.save_results(ident, results)

		return results
		
	def exit_run(self, cause, code=0):
		
		if not self.silent:
			print(f'Exiting due to {cause} (code={code})')
		
		sys.exit(code)
	
	# endregion

@fig.AutoModifier('cls-run')
class OnCluster(Run):
	
	def startup(self):
		
		super().startup()
		
		save_dir = self.save_dir
		
		if save_dir is not None and 'JOBDIR' in os.environ:
			jobdir = os.environ['JOBDIR']
			cname = 'checkpoints{}.txt'.format(os.environ['PROCESS_ID'])
			
			if cname not in os.listdir(jobdir):
				
				# register job
				if 'JOB_REGISTRY_PATH' in os.environ:
					rpath = os.environ['JOB_REGISTRY_PATH']
					reg = load_yaml(rpath) if os.path.isfile(rpath) else []
					reg.append({
						'timestamp': get_now(),
						
						'id': os.environ['JOB_ID'].split('#')[-1],
						'num': int(os.environ['JOB_NUM']),
						'proc': int(os.environ['PROCESS_ID']),
						
						'host': socket.gethostname(),
						
						'run': save_dir,
						'job': jobdir,
					})
					save_yaml(reg, rpath)
					
				with open(os.path.join(jobdir, cname), 'w') as f:
					f.write(os.path.basename(save_dir))
				print('[Saved checkpoint dir for restarts]')
		
		
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
			

		