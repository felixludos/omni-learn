from .imports import *
from ..core import ToolKit, Event
from ..abstract import (AbstractTrainer, AbstractPlanner, AbstractBatch, AbstractDataset, AbstractEvent)

from ..util import DynamicMeter, IntervalMeter, Meter



class Checkpointer(Event):
	def __init__(self, saveroot: Path, *, freq: int = None, skip_0: bool = True, **kwargs):
		super().__init__(**kwargs)
		self._saveroot = saveroot
		self._freq = freq
		self._skip_0 = skip_0
		self._subject = None
		self._last_checkpoint = None
		self._savepath = None


	def settings(self) -> Dict[str, Any]:
		return {'freq': self._freq, 'saveroot': str(self._saveroot.absolute().expanduser())}


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
		if self._freq and batch['num_iterations'] % self._freq == 0 and (not self._skip_0 or batch['num_iterations'] > 0):
			path = self.checkpoint_subject(f'ckpt_{str(batch["num_iterations"]).zfill(6)}')
			print(f' checkpointed to {path}')


	def end(self, last_batch: AbstractBatch = None) -> None:
		path = self.checkpoint_subject(f'ckpt_{str(last_batch["num_iterations"]+1).zfill(6)}')
		print(f' checkpointed to {path}')



class ReporterBase(AbstractEvent):
	def setup(self, trainer, planner, batch_size):
		return self
	
	def step(self, batch):
		pass

	def end(self, last_batch=None):
		pass


class Pbar_Reporter(ReporterBase):
	def __init__(self, *, print_metrics: Iterable[str] = (), print_objective: bool = True, show_pbar: bool = True,
				 unit: str = 'samples', include_epochs: bool = None, ema_halflife: float = 5, ema_alpha: float = None, **kwargs):
		unit = unit.lower()
		assert unit.startswith('s') or unit.startswith('i') or unit.startswith('e'), f'unit should be "samples" or "iterations" or "epochs", not "{unit}"'
		if include_epochs is None and unit.startswith('e'):
			include_epochs = False
		super().__init__(**kwargs)
		self._show_pbar = show_pbar
		self._bar_unit = unit
		self._print_metrics = list(print_metrics)
		self._pbar = None
		self._meter_ema_alpha = ema_alpha
		self._meter_ema_halflife = ema_halflife
		self._include_epochs = include_epochs
		self._itrs_per_epoch = None
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

		dataset_size = getattr(planner, '_dataset_size', None) # TODO: clean up
		if dataset_size is None and (self._bar_unit.startswith('e') or self._include_epochs):
			raise ValueError('dataset size must be known to include epochs in progress bar')
		if dataset_size is not None:
			self._itrs_per_epoch = dataset_size / batch_size
		
		total = planner.expected_samples(batch_size) if self._bar_unit.startswith('s') else total_iterations
		if self._bar_unit.startswith('e'):
			assert dataset_size is not None, 'dataset size must be known to include epochs in progress bar'
			total = dataset_size / batch_size

		unit = 'x' if self._bar_unit.startswith('s') \
			else 'ep' if self._bar_unit.startswith('e') else 'it'
		
		bar_format = "{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]" if self._bar_unit.startswith('e') else "{l_bar}{bar}{r_bar}"

		if self._show_pbar:
			import tqdm
			pbar_type = tqdm.notebook.tqdm if where_am_i() == 'jupyter' else tqdm.tqdm
			self._pbar = pbar_type(total=total, unit=unit, bar_format=bar_format)
			# self._pbar = pbar_type(total=total, unit=unit, unit_scale=1. / self._itrs_per_epoch if self._bar_unit.startswith('e') else None)

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
			inc = batch.size if self._bar_unit.startswith('s') else 1
			if self._bar_unit.startswith('e'):
				inc = 1 / self._itrs_per_epoch
			self._pbar.update(inc)
		return super().step(batch)


	_meter_width = 5
	def _default_pbar_desc(self, batch: AbstractBatch) -> str:

		meters = {self._objective_key: self._objective_meter} if self._objective_meter is not None else {}
		meters.update(self._meters)
		return ', '.join([f'{key}={fixed_width_format_positive(meter.current, self._meter_width)}' for key, meter in meters.items()])
	

	def _default_pbar_post(self, batch: AbstractBatch) -> str:
		if self._itrs_per_epoch is not None and (self._include_epochs or self._include_epochs is None):
			epoch = batch['num_iterations'] / self._itrs_per_epoch
			return f'{epoch:.2f} epochs'


	def end(self, last_batch: AbstractBatch = None) -> None:
		if self._pbar is not None:
			self._pbar.close()


	def report_metrics(self, metrics: Dict[str, Any], iteration: int, *, key_fmt='train/{key}') -> None:
		pass


class WandB_Monitor(Event):
	def __init__(self, *, freqs: Dict[str, int] = None, project_name: str = None, use_wandb: bool = None, wandb_dir: Union[str, Path] = None, max_imgs: int = 12, details: str = None, **kwargs):
		super().__init__(**kwargs)
		self._project_name = project_name
		self._wandb_dir = Path(wandb_dir) if wandb_dir is not None else None
		self._details = details
		self._freqs = {key: freq for key, freq in freqs.items() if freq is not None and freq > 0}
		self._use_wandb = (use_wandb or use_wandb is None) and freqs is not None and len(freqs) > 0
		self._max_imgs = max_imgs
		from torchvision.transforms import ToPILImage
		self._to_pil = ToPILImage()


	def settings(self) -> Dict[str, Any]:
		return {'freqs': self._freqs}


	def setup(self, trainer: AbstractTrainer, src: AbstractDataset, *, device: Optional[str] = None) -> Self:
		if self._use_wandb:
			project_settings = trainer.settings
			if self._project_name is None:
				project_name = trainer.name
			else:
				project_name = pformat(self._project_name,
									   {'model': trainer.model, 'optimizer': trainer.optimizer, 'dataset': src,
									   'trainer': trainer, 'settings': project_settings,
										**trainer.environment()})
			if self._details is not None:
				project_name = f'{project_name}-{self._details}'

			try:
				import wandb
				wandb.init(project=project_name, config=project_settings, dir=self._wandb_dir)
			except ImportError:
				if self._use_wandb:
					raise
				print('WARNING: wandb could not be imported, skipping')
				self._use_wandb = False
		return super().setup(trainer, src, device=device)


	def step(self, batch: AbstractBatch) -> None:
		out = super().step(batch)
		if self._use_wandb:
			itr = batch['num_iterations'] + 1
			content = {key: self._format_content(key, batch[key]) for key, freq in self._freqs.items()
					   if freq and itr > 0 and itr % freq == 0 and batch.gives(key)}
			if len(content):
				self.report_metrics(content, itr)
		return out
	
	def _format_content(self, key, raw):
		import wandb, torchvision
		if isinstance(raw, torch.Tensor) and raw.dtype == torch.uint8 and len(raw.size()) == 4:
			n = min(self._max_imgs, raw.size(0))
			nrows = int(n ** 0.5)
			grid = torchvision.utils.make_grid(raw[:n], nrow=nrows)
			return wandb.Image(self._to_pil(grid))
		return raw

	def report_metrics(self, metrics: Dict[str, Any], iteration: int, *, key_fmt='train/{key}') -> None:
		if self._use_wandb and len(metrics):
			import wandb
			wandb.log({key if key_fmt is None else key_fmt.format(key=key): val 
			  			for key, val in metrics.items()}, step=iteration)
	

	def end(self, last_batch: AbstractBatch = None) -> None:
		out = super().end(last_batch)
		if self._use_wandb:
			import wandb
			wandb.finish()
		return out



class EvaluatorBase(Event):
	def __init__(self, metrics: Iterable[str] = None, *, freq: int = None, eval_reporter = None, skip_0: bool = True, eval_src: AbstractDataset = None, show_pbar: bool = True, single_batch: bool = True, batch_size: int = None, eval_batch_size: int = None, prefix: str = 'val', **kwargs):
		if metrics is None:
			metrics = []
		super().__init__(**kwargs)
		self._freq = freq
		self._single_batch = single_batch
		self._skip_0 = skip_0
		if isinstance(metrics, dict):
			metrics = [key for key, val in metrics.items() if val]
		self._metrics = list(metrics) # TODO: make sure to gauge transform these
		self._batch_size = batch_size
		self._eval_batch_size = eval_batch_size
		self._prefix = prefix
		self._eval_src = eval_src
		self._trainer = None
		self._gadgtry = None
		self._reporter = eval_reporter
		self._show_pbar = show_pbar

	def gizmos(self):
		yield from (f'{self._prefix}_{key}' for key in self._metrics)
		yield from super().gizmos()

	def grab_from(self, ctx, gizmo):
		if gizmo.startswith(f'{self._prefix}_'):
			key = gizmo[len(f'{self._prefix}_'):]
			metrics = self.run(True, metrics=[key])
			return metrics[key].avg
		return super().grab_from(ctx, gizmo)


	def setup(self, trainer: AbstractTrainer, src: AbstractDataset, *, device: Optional[str] = None) -> Self:
		if self._reporter is None:
			self._reporter = trainer.reporter
		self._gadgtry = tuple(trainer.gadgetry())
		self._trainer = trainer
		if self._eval_src is None:
			# assert isinstance(src, AbstractEvaluatableDataset), f'{src} must be an instance of AbstractEvaluatableDataset'
			self._eval_src = src.as_eval()
		self._eval_src.prepare(device=device)
		return super().setup(trainer, src, device=device)

	
	def step(self, batch: AbstractBatch) -> None:
		itr = batch['num_iterations'] + 1
		if self._freq and itr % self._freq == 0 and (not self._skip_0 or itr > 0):
			metrics = self.run(self._single_batch)
			if self._reporter is not None and metrics:
				self._report_metrics(metrics, itr, key_fmt=f'{self._prefix}/{{key}}')


	def end(self, last_batch: AbstractBatch = None) -> None:
		out = self.run(False) if last_batch is not None else None
		self._eval_src = None
		self._gadgtry = None
		if self._reporter is not None and out:
			self._report_metrics(out, last_batch['num_iterations'] + 1, key_fmt=f'final/{{key}}')
		if out:
			nums = {key: (val.avg, val.std, val.max, val.min, val.count) for key, val in out.items() if isinstance(val, self._Meter)}
			missing = set(out) - set(nums)
			if missing:
				print(f'Keys not included here: {missing}')
			print(tabulate([(key, *nums[key]) for key in sorted(nums)], headers=['key', 'avg', 'std', 'max', 'min', 'count']))
		self._reporter = None
		return out
	
	def _report_metrics(self, metrics, iteration, key_fmt='val/{key}'):
		if metrics:
			reportable = {key: score.avg if isinstance(score, self._Meter) else score for key, score in metrics.items()}
			if len(reportable):
				self._reporter.report_metrics(reportable, iteration, key_fmt=key_fmt)
		return metrics


	_Meter = Meter
	def run(self, single_batch: bool = True, metrics: Iterable[str] = None, **kwargs) -> Dict[str, float]:
		if metrics is None:
			metrics = self._metrics
		if len(metrics) == 0 or self._eval_src is None:
			return
		self._trainer.eval()
		metrics = {key: self._Meter() for key in metrics}
		with torch.no_grad():
			for batch in self._eval_src.iterate(self._batch_size, *self._gadgtry, show_pbar=self._show_pbar and not single_batch, shuffle=single_batch, **kwargs):
				self._run_step(metrics, batch)
				if single_batch:
					break
		self._trainer.train()
		return metrics
	
	def _as_number(self, val: Any) -> float:
		if isinstance(val, torch.Tensor) and val.numel() == 1:
			return val.item()
		if isinstance(val, np.ndarray) and val.size == 1:
			return val.item()
		return val

	def _run_step(self, metrics, batch):
		for key in self._metrics:
			if isinstance(metrics[key], self._Meter) and batch.gives(key):
				val = self._as_number(batch[key])
				if isinstance(val, (int, float)):
					metrics[key].mete(val, n=batch.size)
				else:
					metrics[key] = val
		return metrics


