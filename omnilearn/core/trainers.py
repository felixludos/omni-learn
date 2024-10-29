from .imports import *
from .abstract import AbstractMachine, AbstractModel, AbstractDataset, AbstractOptimizer, AbstractTrainer, AbstractEvent, AbstractPlanner, AbstractReporter
from omniply.apps.training import DynamicTrainerBase as _DynamicTrainerBase
from .events import ReporterBase



class TrainerBase(_DynamicTrainerBase):
	_Reporter = ReporterBase
	def __init__(self, model: AbstractModel, optimizer: AbstractOptimizer, *, 
			  reporter: AbstractEvent = None, env: Dict[str, AbstractMachine] = None, 
			  device: str = None, **kwargs):
		if reporter is None:
			reporter = self._Reporter()
		if env is None:
			env = {}
		super().__init__(**kwargs)
		self._model = model
		self._dataset = None
		self._optimizer = optimizer
		self._reporter = reporter
		self._device = device
		self._env = env
		self.extend(env.values())


	@property
	def name(self) -> str:
		src = 'unknown' if self._dataset is None else self._dataset.name
		return f'{self._model.name}_{src}'
	

	@property
	def settings(self) -> Dict[str, Any]:
		assert 'model' not in self._env and 'optimizer' not in self._env, f'env cannot contain "model" or "optimizer": {self._env}'
		return {'model': self._model.settings(), 'optimizer': self._optimizer.settings(), 
		  **{k: v.settings() for k, v in self._env.items()}}


	@property
	def model(self) -> AbstractModel:
		return self._model
	

	@property
	def optimizer(self) -> AbstractOptimizer:
		return self._optimizer


	def all_indicators(self) -> Dict[str, int]:
		raise NotImplementedError
	

	def gadgetry(self) -> Iterator[AbstractGadget]:
		yield self._model
		yield self._optimizer
		yield from super().gadgetry()


	def _setup_fit(self, src: AbstractDataset, *, device: str = None, **settings: Any) -> AbstractPlanner:
		if device is None:
			device = self._device
		
		src = src.load(device=device)

		planner = self._planner.setup(src, **settings)

		self._dataset = src
		self._model.prepare(src, device=device)
		self._optimizer.setup(self._model, device=device)

		return planner


	def _end_fit(self, batch: Batch) -> None:
		pass


	def fit_loop(self, src: AbstractDataset, **settings):
		planner = self._setup_fit(src, **settings)

		batch_size = src.suggest_batch_size() if self._batch_size is None else self._batch_size

		reporter = self._reporter.setup(self, planner, batch_size)

		batch = None
		for info in planner.generate(batch_size):
			batch = self._Batch(info, planner=planner).include(src, reporter).extend(tuple(self.gadgetry()))

			# Note: this runs the optimization step before yielding the batch
			yield self.learn(batch)
			reporter.step(batch)

			if self._terminate_fit(batch):
				break

		self._end_fit(batch)
		reporter.end(batch)


	def learn(self, batch: Batch) -> Batch:
		self._optimizer.step(batch)
		return batch



class Checkpointable(TrainerBase):
	def __init__(self, checkpoint_freq: int = None, **kwargs):
		super().__init__(**kwargs)
		self._my_now = datetime.now().strftime('%Y%m%d_%H%M%S')
		self._checkpoint_freq = checkpoint_freq


	def checkpoint_step(self, batch: Batch) -> Batch:
		itr = batch['iteration']
		if self._checkpoint_freq is not None and itr % self._checkpoint_freq == 0 and itr > 0:
			self.checkpoint(self._create_checkpoint_path(batch))
	

	_max_digits = 6
	def _create_checkpoint_path(self, batch: Batch) -> Path:
		cand = Path(f'{self.name}_{self._my_now}_{str(batch["iteration"]).zfill(self._max_digits)}.pt')
		assert not cand.exists(), f'checkpoint already exists: {cand}'
		return cand


	def checkpoint(self, path: Path = None) -> None:
		outpath = path
		path = None
		ckpt = {
			'model': self._model.checkpoint(path),
			'optimizer': self._optimizer.checkpoint(path),
			'dataset': self._dataset.checkpoint(path),
			**{k: v.checkpoint(path) for k, v in self._env.items()}
		}
		if path is None:
			return ckpt
		torch.save(ckpt, outpath)



