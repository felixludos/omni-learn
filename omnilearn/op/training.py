
import sys
from pathlib import Path
from tqdm import tqdm

from omnibelt import unspecified_argument

import omnifig as fig

from .. import util

# from .loading import load_config, load_records, setup_logging, setup_records, \
# 	wrap_datasets, wrap_transaction, save_checkpoint
from .loading import respect_config
from .evaluation import evaluate
from .clock import Freq, Reg
from .framework import Visualizable



class RunEvent(Freq):
	def check(self, tick, info=None):
		return self.freq is not None and super().check(tick, info=info)
	
	def purge(self):
		pass

@fig.Component('run/epoch')
class Epoch(RunEvent):
	def __init__(self, A, **kwargs):
		step_limit = A.pull('step-limit', None)
		inline = A.pull('inline', False)
		pbar = None
		if inline:
			pbar = A.pull('pbar', None)
		mode = A.pull('mode', 'val')
		print_results = A.pull('print_metric', True)
		fixed_loader_seed = A.pull('loader-seed', None)

		dataset = A.pull('dataset', None, ref=True)
		model = A.pull('model', None, ref=True)
		records = A.pull('records', None, ref=True)
		
		super().__init__(A, **kwargs)
		
		self.mode = mode
		self.loader_seed = fixed_loader_seed
	
		self.dataset = dataset
		self.model = model
		self.records = records
	
		self.step_limit = step_limit
		# if self.step_limit is not None:
		# 	raise NotImplementedError
		
		self.print_results = print_results
		self.pbar = pbar
		
	def purge(self):
		self.dataset = None
		self.model = None
		self.records = None
		super().purge()
		
	def get_mode(self):
		return self.get_name()
	
	def set_mode(self, mode):
		self.mode = mode
	
	@staticmethod
	def pre_epoch(mode, dataset, model, records=None):
		
		dataset.switch_to(mode)
		model.switch_to(mode)
		if records is not None:
			records.switch_to(mode)
			
			if mode not in records['total_epochs']:
				records['total_epochs'][mode] = 0
			if mode not in records['total_steps']:
				records['total_steps'][mode] = 0
			if mode not in records['total_samples']:
				records['total_samples'][mode] = 0
			
	@staticmethod
	def post_epoch(mode, dataset, model, records=None):
		
		if records is not None:
			if mode not in records['total_epochs']:
				records['total_epochs'][mode] += 1
		
			records['total_epochs'][mode] += 1

		# if self.print_results:
		# 	metric = model.get_metric()
		# 	print('')
	
	@staticmethod
	def epoch_step(batch, model, records=None):
		
		out = model.step(batch)
		if records is not None:
			mode = records.get_mode()
			records['total_samples'][mode] += batch.size(0)
			records['total_steps'][mode] += 1
			
		return out

	@staticmethod
	def epoch_end(dataset, model, records=None, batch=None, out=None):
		
		if out is not None and records is not None:
			records.step()
			if isinstance(model, Visualizable):
				model.visualize(out, records)
		
		name = model.optim_metric
		meter = model.get_metric()
		
		mode = model.get_mode()
		title = mode
		if records is not None:
			ep = records['total_epochs'].get(mode, 0) + 1
			title = f'{mode} ({ep})'
		print(f'{title} {name}: {meter.avg:.4f} (min={meter.min:.4f}, max={meter.max:.4f})')
		print()
		

	def run_epoch(self, dataset=unspecified_argument, model=unspecified_argument, records=unspecified_argument,
	              pre_epoch=unspecified_argument, step_fn=unspecified_argument,
	              epoch_end=unspecified_argument, post_epoch=unspecified_argument):
		
		# region fill in missing args
		
		if pre_epoch is unspecified_argument:
			pre_epoch = self.pre_epoch
		if step_fn is unspecified_argument:
			step_fn = self.epoch_step
		if epoch_end is unspecified_argument:
			epoch_end = self.epoch_end
		if post_epoch is unspecified_argument:
			post_epoch = self.post_epoch
		
		if dataset is unspecified_argument:
			dataset = self.dataset
		if model is unspecified_argument:
			model = self.model
		if records is unspecified_argument:
			records = self.records
		
		# endregion
		
		pre_epoch(self.mode, dataset, model, records)
		
		loader = dataset.get_loader(infinite=False, seed=self.loader_seed)
		N = len(loader)
		if self.step_limit is not None and N > self.step_limit:
			N = self.step_limit
		loader = enumerate(loader)
		if self.pbar is not None:
			loader = self.pbar(loader, limit=N)
		
		if records is not None:
			ep = records['total_epochs'].get(self.mode,0) + 1
			title = f'{self.mode}:{ep}'
		else:
			title = self.mode
		
		batch, out = None, None
		for i, batch in loader:
			out = step_fn(batch, model, records)
			
			if self.pbar is not None:
				progress = model.get_description()
				self.pbar.set_description(f'{title} {progress}')
			
			if self.step_limit is not None and i >= self.step_limit:
				break
		
		if self.pbar is not None:
			self.pbar.reset()
		
		epoch_end(dataset, model, records, batch, out)
		
		post_epoch(self.mode, dataset, model, records)
		
		return batch, out
		
	
	def activate(self, tick=None, info=None):
		
		dataset, model, records = None, None, None
		
		if info is not None:
			dataset = info.get_data() if self.dataset is None else self.dataset
			model = info.get_model() if self.model is None else self.model
			records = info.get_records() if self.records is None else self.records

		if records is not None:
			records.set_step(tick)
		
		prev_mode = model.get_mode()
		
		if self.pbar is not None:
			self.pbar.pause()
		
		self.run_epoch(dataset, model, records)
		
		if self.pbar is not None:
			self.pbar.unpause()
		
		dataset.switch_to(prev_mode)
		model.switch_to(prev_mode)
		if records is not None:
			records.switch_to(prev_mode)
	
@fig.Component('run/checkpoint')
class Checkpointer(RunEvent):
	def __init__(self, A, **kwargs):
		
		path = A.pull('save-root', '<>root', '<>path', None)
		
		limit = A.pull('keep-only', '<>limit', None)
		
		# metric = A.pull('track-best', None)
		
		dataset = A.pull('dataset', None, ref=True)
		model = A.pull('model', None, ref=True)
		records = A.pull('records', None, ref=True)
		clock = A.pull('clock', None, ref=True)
		
		super().__init__(A, **kwargs)
		
		self.root = Path(path)
		self.limit = limit
		if self.limit is not None:
			raise NotImplementedError
		# self.best_metric = metric
		# if self.best_metric is not None:
		# 	raise NotImplementedError
	
		self.dataset = dataset
		self.model = model
		self.records = records
		self.clock = clock
	
	def purge(self):
		self.dataset = None
		self.model = None
		self.records = None
		self.clock = None
	
	def _limit_checkpoints(self, root):
		if self.limit is not None:
			raise NotImplementedError # TODO
	
	def _save_best(self):
		if self.best_metric is not None:
			raise NotImplementedError # TODO
	
	@staticmethod
	def save_checkpoint(path, dataset=None, model=None, records=None, clock=None):

		path = Path(path)
		path.mkdir(exist_ok=True)

		if dataset is not None:
			dataset.checkpoint(path)
		if model is not None:
			model.checkpoint(path)
		if records is not None:
			records.checkpoint(path)
		if clock is not None:
			clock.checkpoint(path)
	
	def create_path(self, num, root=None):
		if root is None:
			root = self.root
	
		if root is None:
			pass
			
		path = Path(root) / f'ckpt{num}'
		return path
	
	def checkpoint(self, dataset=None, model=None, records=None, clock=None, root=None, num=None):
		
		path = self.create_path(num=num, root=root)
		if path is None:
			return
		path.mkdir(exist_ok=True)
		
		if dataset is None:
			dataset = self.dataset
		if model is None:
			model = self.model
		if records is None:
			records = self.records
		if clock is None:
			clock = self.clock
		
		if records is not None:
			records.prep_checkpoint(num)
		
		return self.save_checkpoint(path, dataset, model, records, clock)
		
	def activate(self, tick, info=None):
		assert info is not None
		
		root = info.get_path() if self.root is None else self.root
		
		if root is None:
			return
		
		dataset = info.get_dataset() if self.dataset is None else self.dataset
		model = info.get_model() if self.model is None else self.model
		records = info.get_records() if self.records is None else self.records
		clock = info.get_clock() if self.clock is None else self.clock
		
		self.checkpoint(dataset=dataset, model=model, records=records, clock=clock, root=root, num=tick)
		
		self._limit_checkpoints(root)
		
		
@fig.Component('run/viz')
class VizStep(RunEvent):
	def __init__(self, A, **kwargs):
		
		model = A.pull('model', None, ref=True)
		records = A.pull('records', None, ref=True)
		
		super().__init__(A, **kwargs)
		
		self.model = model
		self.records = records
		
	def activate(self, tick, info=None):
		
		assert info is not None, 'no info provided'
		
		model = info.get_model() if self.model is None else self.model
		records = info.get_records() if self.records is None else self.records
		
		out = info.get_output()
	
		if isinstance(model, Visualizable):
			if records is not None:
				records.set_step(tick)
			model.visualize(out, records)


@fig.Component('run/print')
class PrintStep(RunEvent):
	def activate(self, tick, info=None):
		if info is not None:
			desc = info.get_description(tick)
			print(desc)
			sys.stdout.flush()


@fig.Script('train', description='Train new/existing models')
def iterative_training(A=None, run=None):
	'''
	This is the entry for the training script for new or existing runs.
	Existing runs (or models) can be specified using "path", "load",
	or "resume" with the run name
	'''
	
	respect_config(A)

	if run is None:
		assert A is not None, 'either run or A must not be None'
		run = fig.run('load-run', A)
	
	A = run.get_config()
	
	run.take_steps(complete=True)
	
	include_eval = A.pull('include-eval', True)
	if include_eval:
		evaluate(A=None, run=run)
	
	print(f'Completed {run.get_name()}')
	
	return run


