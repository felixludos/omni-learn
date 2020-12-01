
import sys, os
from pathlib import Path
from tqdm import tqdm
import time

import torch

import omnifig as fig

from .framework import Recordable, Evaluatable, Visualizable

from .. import util

# from .loading import load_config, load_records, setup_logging, setup_records, \
# 	wrap_datasets, wrap_transaction, save_checkpoint
from .loading import respect_config
from .model import load_model
from .data import load_data
from .evaluation import eval_model
from .clock import Alert, Freq, Reg

class Run_Event(Reg, Freq):
	def check(self, tick, info=None):
		return self.freq is not None and super().check(tick, info=info)
	
@fig.Component('run/epoch')
class CompleteEpoch(Run_Event):
	def __init__(self, A, **kwargs):
		epoch_limit = A.pull('epoch-limit', None)
		pbar = A.pull('pbar', None)
		super().__init__(A, **kwargs)
	
		self.epoch_limit = epoch_limit
		
		self.pbar = pbar
		
	def get_mode(self):
		return self.get_name()
	
	@classmethod
	def epoch_step(cls, mode, batch, model, records):
		
		B = batch.size(0)
		if mode in records['total_samples']:
			records['total_samples'][mode] += B
		
		out = model.test(batch)
		if 'loss' in out:
			records.stats.update('loss', out.loss.detach())
			
		return out

	@classmethod
	def run_epoch(cls, mode, loader, model, records=None, logger=None, step_fn=None):
		
		if step_fn is None:
			step_fn = cls.epoch_step
		
		out = None
		for batch in loader:
			out = step_fn(mode, batch, model, records)
		
		if out is not None and logger is not None:
			logger.step(records, fmt='{}/{}'.format('{}', mode))
			if isinstance(model, Visualizable):
				model.visualize(out, logger)
	
	def activate(self, tick, info=None):
		assert info is not None
		
		mode = self.get_mode()
		
		dataset = info.get_dataset(mode)
		if dataset is None:
			return
		
		model = info.get_model()
		model.switch_mode(mode)
		
		records = info.get_records()
		records.switch_mode(mode)
		
		loader = dataset.to_loader(infinite=False)
		
		logger = info.get_logger()
		
		self.run_epoch(mode, loader, model, records=records, logger=logger)
	
@fig.Component('run/checkpoint')
class Checkpointer(Run_Event):
	def __init__(self, A, **kwargs):
		
		path = A.pull('save-path', '<>path', None)
		
		limit = A.pull('ckpt-limit', '<>keep-last', None)
		
		best = A.pull('track-best', False)
		
		super().__init__(A, **kwargs)
		
		self.path = Path(path)
		self.limit = limit
		self.best = best
	
	def _limit_checkpoints(self, root):
		if self.limit is not None:
			
			
			pass
		
		
	
	
	def activate(self, tick, info=None):
		assert info is not None
		
		root = info.get_path() if self.path is None else self.path
		if root is None:
			return
		records = info.get_records()
		clock = info.get_clock()
		
		num = clock.get_time()
		path = root / f'ckpt{num}'
		path.mkdir()
		
		self._limit_checkpoints(root)
		
		pass

@fig.Script('train', description='Train new/existing models')
def iterative_training(A=None, run=None):
	'''
	This is the entry for the training script for new or existing runs.
	Existing runs (or models) can be specified using "path", "load",
	or "resume" with the run name
	'''
	#####################
	# region Loading
	#####################

	respect_config(A)

	if run is None:
		assert A is not None, 'either run or A must not be None'
		A.push('run._type', 'run', overwrite=False)
		run = A.pull('run')
	
	A = run.get_config()

	safe_config = A.pull('safe_config', False)

	if safe_config:
		A.begin()
	
	datasets = run.get_datasets()
	model = run.get_model()

	if safe_config:
		A.abort()

	logger = run.get_logger()
	
	run.prepare()
	
	loaders = run.get_loaders()
	
	# endregion
	#######################
	# region Smart Defaults
	#######################
	
	if 'train' not in datasets:
		raise Exception(f'No training dataset found (how did this happen?)')
		
	for key in ['train', 'val', 'test']:
		if key in datasets:
			if datasets[key] is None:
				print(f'{key} is None')
			else:
				print(f'{key}data len={len(datasets[key])}, {key}loader len={len(loaders[key])}')
	
	trainloader = loaders['train']
	epoch_len = len(trainloader)
	
	tau = A.push('training.stats.tau', max(0.01, min(100/epoch_len, 0.1)), overwrite=False)
	util.set_default_tau(tau)
	if isinstance(model, Recordable):
		model.stats.set_tau(tau)
	A.push('output.print_freq', min(max(20, epoch_len // 40), 200), overwrite=False)
	A.push('output.log_freq', min(max(20, epoch_len // 40), 200), overwrite=False)
	
	epochs = A.pull('training.epochs', 10)
	step_limit = A.push('training.step_limit', epochs * epoch_len, overwrite=False)

	expected_epochs = A.push('expected_epochs', step_limit//epoch_len, overwrite=False)

	no_val = A.pull('training.no_val', False)
	A.push('training.val_freq', None if no_val else epoch_len, overwrite=False)

	inline = A.pull('inline', False)
	pbar = tqdm if inline else None
	
	# endregion
	#####################
	# region Model
	#####################
	
	print(model)
	print(model.optim)
	if hasattr(model, 'scheduler'):
		print(model.scheduler)
	print('Model has {} parameters'.format(util.count_parameters(model)))
	
	sys.stdout.flush()
	
	# endregion
	#####################
	# region Run Training
	#####################
	
	remaining_steps = step_limit - run.get_total_steps()
	
	if remaining_steps > 0:
		print(f'Training for {remaining_steps} steps')
		# run.prepare(model, trainloader)
		run.continuous(pbar=pbar)
		print('Training complete.')
	
	else:
		print('No training')
	
	# endregion
	#####################
	# region Run Evaluation
	#####################

	include_eval = A.pull('eval-after-training', True)

	if include_eval and isinstance(model, Evaluatable):
		eval_model(A, run=run)
		
	# endregion
	
	return run


