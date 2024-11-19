from .imports import *
from .abstract import AbstractMachine, AbstractModel, AbstractDataset, AbstractOptimizer, AbstractTrainer, AbstractEvent, AbstractPlanner, AbstractReporter
from omniply.apps.training import DynamicTrainerBase as _DynamicTrainerBase
from .events import ReporterBase



class TrainerBase(_DynamicTrainerBase):
	_Reporter = ReporterBase
	def __init__(self, model: AbstractModel, optimizer: AbstractOptimizer, *, 
			  reporter: AbstractEvent = None, env: Dict[str, AbstractMachine] = None, 
			  events: Iterable[AbstractEvent] = None,
			  device: str = None, **kwargs):
		if reporter is None:
			reporter = self._Reporter()
		if env is None:
			env = {}
		if events is None:
			events = [e for e in env.values() if isinstance(e, AbstractEvent)]
		super().__init__(**kwargs)
		self._model = model
		self._dataset = None
		self._optimizer = optimizer
		self._reporter = reporter
		self._device = device
		self._env = env
		self._events = events
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
	

	@property
	def reporter(self) -> AbstractReporter:
		return self._reporter


	def all_indicators(self) -> Iterator[str]:
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
		for e in self._events:
			e.setup(self, src, device=device)

		return planner


	def _end_fit(self, batch: Batch) -> None:
		for e in self._events:
			e.end(batch)


	# def loop(self, src: AbstractDataset, **settings: Any) -> Iterator[Batch]:
	# 	return self.fit_loop(src, **settings)


	def fit_loop(self, src: AbstractDataset, **settings):
		planner = self._setup_fit(src, **settings)

		batch_size = src.suggest_batch_size() if self._batch_size is None else self._batch_size

		reporter = self._reporter.setup(self, planner, batch_size)

		batch_cls = self._Batch or getattr(src, '_Batch', None) or Batch
		
		batch = None
		for info in planner.generate(batch_size):
			batch = batch_cls(info, planner=planner).include(src, reporter).extend(tuple(self.gadgetry()))

			# Note: this runs the optimization step before yielding the batch
			terminate = self.learn(batch)
			
			yield batch

			if terminate:
				break

		self._end_fit(batch)
		reporter.end(batch)


	def learn(self, batch: Batch) -> bool:
		self._optimizer.step(batch)
		for e in self._events:
			e.step(batch)
		self.reporter.step(batch)

		return self._terminate_fit(batch)



class CheckpointableTrainer(TrainerBase):
	def __init__(self, checkpoint_freq: int = None, **kwargs):
		super().__init__(**kwargs)
		self._my_now = datetime.now().strftime('%Y%m%d_%H%M%S')


	@property
	def name(self) -> str:
		return f'{super().name}_{self._my_now}'


	def checkpoint(self, path: Path) -> None:
		path.mkdir()

		self.model.checkpoint(path / 'model')
		self.optimizer.checkpoint(path / 'optimizer')
		if self._dataset is not None:
			self._dataset.checkpoint(path / 'dataset')
		for ident, machine in self._env.items():
			machine.checkpoint(path / ident)

		return path




