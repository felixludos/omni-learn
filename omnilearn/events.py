from .imports import *
from .abstract import AbstractTrainer, AbstractEvent, AbstractReporter, AbstractPlanner, AbstractBatch, AbstractDataset, AbstractEvaluatableDataset

from .util import fixed_width_format_value, fixed_width_format_positive



class Checkpointer(AbstractEvent):
	def __init__(self, saveroot: Path, *, freq: int = None, skip_0: bool = True, **kwargs):
		super().__init__(**kwargs)
		self._saveroot = saveroot
		self._freq = freq
		self._skip_0 = skip_0
		self._subject = None
		self._last_checkpoint = None
		self._savepath = None


	def setup(self, trainer: AbstractTrainer, src: AbstractDataset, *, device: Optional[str] = None) -> Self:
		self._saveroot.mkdir(exist_ok=True, parents=True)
		self._savepath = self._saveroot / f'{trainer.name}'
		assert not self._savepath.exists(), f'{self._savepath} already exists'
		self._subject = trainer
		return self
	

	def checkpoint_subject(self, name: str) -> None:
		if not self._savepath.exists():
			self._savepath.mkdir()
		
		path = self._savepath / name
		exists = len(list(self._savepath.glob(f'{name}*'))) > 0
		if not exists:
			self._subject.checkpoint(path)
		return path


	def step(self, batch: AbstractBatch) -> None:
		if self._freq is not None and batch['num_iterations'] % self._freq == 0 and (not self._skip_0 or batch['num_iterations'] > 0):
			path = self.checkpoint_subject(f'ckpt_{str(batch["num_iterations"]).zfill(6)}')
			print(f' checkpointed to {path}')


	def end(self, last_batch: AbstractBatch = None) -> None:
		self.checkpoint_subject(f'ckpt_{str(last_batch["num_iterations"]).zfill(6)}')


class ReporterBase(ToolKit, AbstractReporter):
	def setup(self, trainer, planner, batch_size):
		return self
	
	def step(self, batch):
		pass

	def end(self, last_batch=None):
		pass



class Pbar_Reporter(ReporterBase):
	def __init__(self, *, print_metrics: Iterable[str] = (), print_objective: bool = True, show_pbar: bool = True, unit: str = 'samples', ema_halflife: float = 5, ema_alpha: float = None, **kwargs):
		assert unit.startswith('s') or unit.startswith('i'), f'unit should be "samples" or "iterations", not "{unit}"'
		super().__init__(**kwargs)
		self._show_pbar = show_pbar
		self._count_samples = unit.startswith('s')
		self._print_metrics = list(print_metrics)
		self._pbar = None
		self._meter_ema_alpha = ema_alpha
		self._meter_ema_halflife = ema_halflife
		self._objective_key = None
		self._objective_meter = DynamicMeter(alpha=ema_alpha, target_halflife=ema_halflife) if print_objective else None
		self._meters = None


	_max_alpha = 0.01
	def setup(self, trainer: AbstractTrainer, planner: AbstractPlanner, batch_size: int) -> Self:
		total_iterations = planner.expected_iterations(batch_size)

		self._meters = {key: DynamicMeter(alpha=self._meter_ema_alpha, target_halflife=self._meter_ema_halflife) for key in self._print_metrics}

		# self.gauge_apply({'objective': trainer.optimizer.objective})
		self._objective_key = trainer.optimizer.objective
		self._objective_meter.reset()

		total = planner.expected_samples(batch_size) if self._count_samples else total_iterations

		if self._show_pbar:
			import tqdm
			pbar_type = tqdm.notebook.tqdm if where_am_i() == 'jupyter' else tqdm.tqdm
			self._pbar = pbar_type(total=total, unit='x' if self._count_samples else 'it')

		return super().setup(trainer, planner, batch_size)


	def step(self, batch: AbstractBatch) -> None:
		if self._pbar is not None:
			for key, meter in self._meters.items():
				meter.mete(batch[key])
			if self._objective_meter is not None:
				self._objective_meter.mete(batch[self._objective_key])

			# desc = batch.grab('pbar_desc', None)
			# if desc is None:
			# 	desc = self._default_pbar_desc(self._objective_meter.current)
			desc = self._default_pbar_desc(batch)
			if desc is not None:
				self._pbar.set_description(desc, refresh=False)
			post = self._default_pbar_post(batch)
			if post is not None:
				self._pbar.set_postfix_str(post, refresh=False)
			self._pbar.update(batch.size if self._count_samples else 1)
		return super().step(batch)


	_meter_width = 5
	def _default_pbar_desc(self, batch: AbstractBatch) -> str:

		meters = {self._objective_key: self._objective_meter} if self._objective_meter is not None else {}
		meters.update(self._meters)
		return ', '.join([f'{key}={fixed_width_format_positive(meter.current, self._meter_width)}' for key, meter in meters.items()])
	

	def _default_pbar_post(self, batch: AbstractBatch) -> str:
		return None


	def end(self, last_batch: AbstractBatch = None) -> None:
		if self._pbar is not None:
			self._pbar.close()



class WandB_Reporter(ReporterBase):
	def __init__(self, *, project_name: Optional[str] = None, 
				 project_settings: Optional[Dict[str, Any]] = None, use_wandb: bool = None, **kwargs):
		super().__init__(**kwargs)
		self._project_name = project_name
		self._project_settings = project_settings
		self._use_wandb = use_wandb
		self._indicators = {}
		self._indicator_costs = {}


	def setup(self, trainer: AbstractTrainer, planner: AbstractPlanner, batch_size: int) -> Self:
		project_name = self._project_name or trainer.name
		project_settings = self._project_settings or trainer.settings

		indicators = list(trainer.all_indicators())
		self._indicators.update({key: DynamicMeter() for key in indicators})
		self._indicator_costs.update({key: IntervalMeter() for key in indicators})

		try:
			import wandb
			wandb.init(project=project_name, config=project_settings)
		except ImportError:
			if self._use_wandb:
				raise
			print('WARNING: wandb could not be imported, skipping')
			self._use_wandb = False
		return super().setup(trainer, planner, batch_size)


	def step(self, batch: AbstractBatch) -> None:
		out = super().step(batch)
		if self._use_wandb:
			import wandb
			itr = batch['iteration']
			wandb.log({key: batch[key] 
					for key, freq in self._indicators.items() 
					if freq > 0 and itr % freq == 0})
		return out


	def report_validation(self, metrics: Dict[str, float], iteration: int) -> None:
		if self._use_wandb:
			import wandb
			wandb.log({f'val_{key}': val for key, val in metrics.items()}, step=iteration)


	def end(self, last_batch: AbstractBatch = None) -> None:
		out = super().end(last_batch)
		if self._use_wandb:
			import wandb
			wandb.finish()
		return out



class Meter:
	def __init__(self, alpha: float = None, window_size: float = None, **kwargs):
		assert alpha is None or window_size is None, 'cannot specify both alpha and window_size'
		if window_size is not None:
			alpha = self.window_size_to_alpha(window_size)
		super().__init__(**kwargs)
		self._alpha = alpha
		self.reset()


	@staticmethod
	def window_size_to_alpha(window_size: float) -> float:
		assert window_size >= 1, f'window_size {window_size} must be >= 1'
		return 2 / (window_size + 1)
		
		
	@property
	def alpha(self):
		return self._alpha


	def reset(self):
		self.last = None
		self.avg = None
		self.sum = 0.
		self.count = 0
		self.max = None
		self.min = None
		self.smooth = None
		self.var = None
		self.std = None
		self.S = 0.
		self.timestamp = None
	

	def mete(self, val: float, *, n=1):
		if isinstance(val, torch.Tensor):
			val = val.detach()
			val = val.item() if val.numel() == 1 else val.numpy()
		self.timestamp = time.time()
		self.last = val
		self.sum += val * n
		prev_count = self.count
		self.count += n
		self.avg = self.sum / self.count
		delta = val - self.avg
		self.S += delta**2 * n * prev_count / self.count
		self.max = val if self.max is None else np.maximum(self.max, val)
		self.min = val if self.min is None else np.minimum(self.min, val)
		self.var = self.S / self.count
		self.std = np.sqrt(self.var)
		alpha = self.alpha
		if alpha is not None:
			self.smooth = val if self.smooth is None else (self.smooth*(1-alpha) + val*alpha)
		return val
	

	@property
	def current(self):
		return self.last if self.smooth is None else self.smooth
	

	@property
	def estimate(self):
		return self.avg if self.smooth is None else self.smooth


	def __len__(self):
		return self.count
	

	def __float__(self):
		return self.current


	# def join(self, other):
	# 	'''other is another average meter, assumed to be more uptodate (for val)
	# 		this does not mutate other'''
	# 	raise NotImplementedError # an implementation can be found in `old/`: `AverageMeter`



_ln2 = np.log(2)


class DynamicMeter(Meter):
	def __init__(self, alpha: float = None, *, infer_alpha: bool = None, max_window_size: float = 1000, min_window_size: float = 10, wait_steps: int = 5, target_halflife: float = 5, **kwargs):
		'''
		target_halflife: the time in sec
		'''
		if infer_alpha is None:
			infer_alpha = alpha is None
		super().__init__(alpha=alpha, **kwargs)
		assert max_window_size >= min_window_size, f'max_window_size {max_window_size} must be >= min_window_size {min_window_size}'
		self._max_alpha = self.window_size_to_alpha(min_window_size)
		self._min_alpha = self.window_size_to_alpha(max_window_size)
		self._wait_steps = wait_steps
		self._target_halflife = target_halflife
		self._alpha_estimator = Meter(window_size=self._wait_steps) if infer_alpha else None
		self._hanging_tick = None
	

	@property
	def alpha(self):
		if self._alpha_estimator is not None:
			estimate = self._alpha_estimator.estimate
			if estimate is not None:
				return min(max(self._min_alpha, self._alpha_estimator.estimate), self._max_alpha)
		return self._alpha


	def mete(self, val: float, *, n=1):
		val = super().mete(val, n=n)
		if self._alpha_estimator is not None:
			if self._hanging_tick is not None:
				dt = self.timestamp - self._hanging_tick
				alpha = 1 - np.exp(-_ln2 / (self._target_halflife/dt))
				self._alpha_estimator.mete(alpha)
				self._hanging_tick = None
			if self._wait_steps is None or self.count >= self._wait_steps:
				self._hanging_tick = self.timestamp
		return val
	


class IntervalMeter(DynamicMeter):
	def __init__(self, alpha: float = None, window_size: float = None, **kwargs):
		super().__init__(alpha=alpha, window_size=window_size, **kwargs)
		self._last_timestamp = None


	def mete(self, *, n=1):
		now = time.time()
		out = None
		if self._last_timestamp is not None:
			out = super().mete(now - self._last_timestamp, n=n)
		self._last_timestamp = now
		return out



# @fig.component('evaluator')
class EvaluatorBase(AbstractEvent):
	def __init__(self, metrics: Iterable[str] = None, *, freq: int = None, skip_0: bool = True, eval_src: AbstractDataset = None, show_pbar: bool = True, batch_size: int = None, eval_batch_size: int = None, prefix: str = 'val', **kwargs):
		if metrics is None:
			metrics = []
		super().__init__(**kwargs)
		self._freq = freq
		self._skip_0 = skip_0
		self._metrics = list(metrics) # TODO: make sure to gauge transform these
		self._batch_size = batch_size
		self._eval_batch_size = eval_batch_size
		self._prefix = prefix
		self._eval_src = eval_src
		self._gadgtry = None
		self._reporter = None
		self._show_pbar = show_pbar


	def setup(self, trainer: AbstractTrainer, src: AbstractDataset, *, device: Optional[str] = None) -> Self:
		self._reporter = trainer.reporter
		self._gadgtry = tuple(trainer.gadgetry())
		if self._eval_src is None:
			assert isinstance(src, AbstractEvaluatableDataset), f'{src} must be an instance of AbstractEvaluatableDataset'
			self._eval_src = src.as_eval()
		return super().setup(trainer, src, device=device)

	
	def step(self, batch: AbstractBatch) -> None:
		if self._freq is not None and batch['num_iterations'] % self._freq == 0 and (not self._skip_0 or batch['num_iterations'] > 0):
			metrics = self.run()
			if self._reporter is not None:
				self._reporter.report_validation(metrics, batch['num_iterations'])


	def end(self, last_batch: AbstractBatch = None) -> None:
		out = self.run() if last_batch is not None else None
		self._eval_src = None
		self._gadgtry = None
		if self._reporter is not None:
			self._reporter.report_validation(out, last_batch['num_iterations'])
		self._reporter = None
		return out


	_Meter = DynamicMeter
	def run(self) -> Dict[str, float]:
		if len(self._metrics) == 0 or self._eval_src is None:
			return
		metrics = {key: self._Meter() for key in self._metrics if not key.startswith(f'{self._prefix}_')}
		with torch.no_grad():
			for batch in self._eval_src.iterate(self._batch_size, *self._gadgtry, show_pbar=self._show_pbar):
				self._run_step(metrics, batch)
		return {key: score.avg for key, score in metrics.items()}
	

	def _run_step(self, metrics, batch):
		for key in self._metrics:
			metrics[key].mete(batch[key], n=batch.size)




