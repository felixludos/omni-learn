
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
from .clock import Alert, Freq, Reg, Priority, Clock
from .framework import Visualizable



class Run_Event(Reg, Freq, Priority):
	def check(self, tick, info=None):
		return self.freq is not None and super().check(tick, info=info)
	
	def purge(self):
		pass

@fig.Component('run/epoch')
class Epoch(Run_Event):
	def __init__(self, A, **kwargs):
		step_limit = A.pull('step-limit', None)
		pbar = A.pull('pbar', None)
		mode = A.pull('mode', 'val')
		metric = A.pull('metric', None)

		dataset = A.pull('dataset', None, ref=True)
		model = A.pull('model', None, ref=True)
		records = A.pull('records', None, ref=True)
		
		if metric is not None and records is not None:
			records.new(metric)
		
		super().__init__(A, **kwargs)
		
		self.mode = mode
	
		self.dataset = dataset
		self.model = model
		self.records = records
	
		self.step_limit = step_limit
		if self.step_limit is not None:
			raise NotImplementedError
		
		self.pbar = pbar
		
	def purge(self):
		self.dataset = None
		self.model = None
		self.records = None
		
	def get_mode(self):
		return self.get_name()
	
	@staticmethod
	def pre_epoch(mode, dataset, model, records=None):
		
		dataset.switch_mode(mode)
		model.switch_mode(mode)
		if records is not None:
			records.switch_mode(mode)
		
	@staticmethod
	def post_epoch(mode, dataset, model, records=None):
		pass
	
	@staticmethod
	def epoch_step(batch, model, records=None):
		
		out = model.step(batch)
		if records is not None:
			records.increment_samples(batch.size(0))
			
		return out

	@staticmethod
	def epoch_end(dataset, model, records, batch, out):
		
		if out is not None and records is not None:
			records.step()
			if isinstance(model, Visualizable):
				model.visualize(out, records)

	def run_epoch(self, dataset=None, model=None, records=None,
	              pre_epoch=None, step_fn=None, epoch_end=None, post_epoch=None):
		
		if pre_epoch is None:
			pre_epoch = self.pre_epoch
		if step_fn is None:
			step_fn = self.epoch_step
		if epoch_end is None:
			epoch_end = self.epoch_end
		if post_epoch is None:
			post_epoch = self.post_epoch
		
		if dataset is None:
			dataset = self.dataset
		if model is None:
			model = self.model
		if records is None:
			records = self.records
		
		prev_mode = model.get_mode()
		
		pre_epoch(self.mode, dataset, model, records)
		
		loader = enumerate(dataset.to_loader(infinite=False))
		if self.pbar is not None:
			loader = self.pbar(loader)
		
		batch, out = None, None
		for i, batch in loader:
			out = step_fn(batch, model, records)
			
			if records is not None and self.metric is not None and self.metric in out:
				records.update(self.metric, out[self.metric].detach())
			
			if self.step_limit is not None and i >= self.step_limit:
				break
		
		if self.pbar is not None:
			self.pbar.reset()
		
		if records is not None:
			epoch_end(dataset, model, records, batch, out)
		
		post_epoch(self.mode, dataset, model, records)
		
		dataset.switch_to(prev_mode)
		model.switch_to(prev_mode)
		if records is not None:
			records.switch_to(prev_mode)
	
	def activate(self, tick=None, info=None):
		
		dataset, model, records = None, None, None
		
		if info is not None:
			dataset = info.get_data() if self.dataset is None else self.dataset
			model = info.get_model() if self.model is None else self.model
			records = info.get_records() if self.records is None else self.records
		
		self.run_epoch(dataset, model, records)
	
@fig.Component('run/checkpoint')
class Checkpointer(Run_Event):
	def __init__(self, A, **kwargs):
		
		path = A.pull('save-root', '<>root', '.')
		
		limit = A.pull('keep-only', '<>limit', None)
		
		metric = A.pull('track-best', None)
		
		dataset = A.pull('dataset', None, ref=True)
		model = A.pull('model', None, ref=True)
		records = A.pull('records', None, ref=True)
		
		super().__init__(A, **kwargs)
		
		self.path = Path(path)
		self.limit = limit
		if self.limit is not None:
			raise NotImplementedError
		self.best_metric = metric
		if self.best_metric is not None:
			raise NotImplementedError
	
		self.dataset = dataset
		self.model = model
		self.records = records
	
	def purge(self):
		self.dataset = None
		self.model = None
		self.records = None
	
	def _limit_checkpoints(self, root):
		if self.limit is not None:
			raise NotImplementedError
	
	def checkpoint(self, path, dataset=None, model=None, records=None):
		
		path = Path(path)
		path.mkdir(exist_ok=True)
		
		if dataset is None:
			dataset = self.dataset
		if dataset is not None:
			dataset.checkpoint(path)
		
		if model is None:
			model = self.model
		if model is not None:
			model.checkpoint(path)
		
		if records is None:
			records = self.records
		if records is not None:
			records.checkpoint(path)
		
	def activate(self, tick, info=None):
		assert info is not None
		
		root = info.get_path() if self.path is None else self.path
		if root is None:
			return
		
		path = root / f'ckpt{tick}'
		
		dataset = info.get_dataset() if self.dataset is None else self.dataset
		model = info.get_model() if self.model is None else self.model
		records = info.get_records() if self.records is None else self.records
		
		self.checkpoint(path, dataset=dataset, model=model, records=records)
		
		self._limit_checkpoints(root)
		
@fig.Component('run/viz')
class VizStep(Run_Event):
	def __init__(self, A, **kwargs):
		
		super().__init__(A, **kwargs)
		
		self.model = A.pull('model', None, ref=True)
		self.records = A.pull('records', None, ref=True)
		
	def activate(self, tick, info=None):
		
		assert info is not None, 'no info provided'
		
		model = info.get_model() if self.model is None else self.model
		records = info.get_records() if self.records is None else self.records
		
		out = info.get_out()
	
		if isinstance(model, Visualizable):
			model.visualize(out, records)


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
	
	run.continuous()
	
	raise NotImplementedError
	

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
	# A.push('output.print_freq', min(max(20, epoch_len // 40), 200), overwrite=False)
	# A.push('output.log_freq', min(max(20, epoch_len // 40), 200), overwrite=False)
	
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
	
	# remaining_steps = step_limit - run.get_total_steps()
	#
	# if remaining_steps > 0:
	# 	print(f'Training for {remaining_steps} steps')
	# 	# run.prepare(model, trainloader)
	# 	run.continuous(pbar=pbar)
	# 	print('Training complete.')
	#
	# else:
	# 	print('No training')
	
	# endregion
	#####################
	# region Run Evaluation
	#####################

	include_eval = A.pull('eval-after-training', True)

	if include_eval and isinstance(model, Evaluatable):
		eval_model(A, run=run)
		
	# endregion
	
	return run


