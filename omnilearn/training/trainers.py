from .imports import *
from ..core import Batch, ToolKit, Structured
from ..abstract import AbstractMachine, AbstractModel, AbstractDataset, AbstractOptimizer, AbstractEvent, AbstractPlanner, AbstractReporter
from omniply.apps.training import DynamicTrainerBase as _DynamicTrainerBase
from ..mixins import Prepared
from .events import ReporterBase



class TrainerBase(Prepared, _DynamicTrainerBase):
	_Reporter = ReporterBase
	def __init__(self, model: AbstractModel, optimizer: AbstractOptimizer, *, 
			  reporter: AbstractEvent = None, env: Dict[str, AbstractMachine] = None, 
			  events: Dict[str, AbstractEvent] = None,
			  device: str = None, **kwargs):
		if reporter is None:
			reporter = self._Reporter()
		if env is None:
			env = {}
		if events is None:
			events = {}
		super().__init__(**kwargs)
		self._model = model
		self._dataset = None
		self._optimizer = optimizer
		self._reporter = reporter
		self._device = device
		self._env = env
		env.update(events)
		self._events = events
		self.extend(env.values())


	@property
	def name(self) -> str:
		src = 'unknown' if self._dataset is None else self._dataset.name
		return f'{self._model.name}_{src}'


	def environment(self) -> Dict[str, Any]:
		'''prepare the dataset for training'''
		return self._env.copy()


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


	def gadgetry(self) -> Iterator[AbstractGadget]:
		yield self._model
		yield self._optimizer
		yield from super().gadgetry()


	def _prepare(self, *, device: str = None):
		"""
		Only run once, only uses dataset implicitly - eg through gears. Generally this is not the top level

		use setup() instead
		"""
		self._model.prepare(device=device)
		print(self._model)
		self._optimizer.setup(self._model)
		self._optimizer.prepare(device=device)
		for e in self._env.values():
			e.prepare(device=device)
		return self


	_System = Structured
	def setup(self, src: AbstractDataset, *, device: str = None):
		self._dataset = src
		if device is None:
			device = self._device
		src = src.load(device=device)
		system = self._System(src, *self.gadgetry())
		system.mechanize() # sync for gears and spaces
		self.prepare(device=device)
		return system


	def _setup_fit(self, src: AbstractDataset, *, device: str = None, **settings: Any) -> AbstractPlanner:
		for e in self._events.values():
			e.setup(self, src, device=device)
		return self._planner.setup(src, **settings)


	def _end_fit(self, batch: Batch) -> None:
		for e in self._events.values():
			e.end(batch)


	def loop(self, batch_size: int = None, *, system: Structured = None, planner: AbstractPlanner = None):
		if batch_size is None:
			assert self._batch_size is not None, 'batch_size must be provided if not set'
			batch_size = self._batch_size
		if planner is None:
			assert self._dataset is not None, 'planner must be provided if dataset is not set'
			planner = self._planner.setup(self._dataset)

		batch_cls = self._Batch or getattr(self._dataset, '_Batch', None) or Batch
		for info in planner.generate(batch_size):
			batch = batch_cls(info, planner=planner)
			if system is not None:
				batch.include(system)
			yield batch


	def fit_loop(self, src: AbstractDataset, *, batch_size: int = None, **settings):
		if batch_size is None:
			batch_size = src.suggest_batch_size() if self._batch_size is None else self._batch_size

		system = self.setup(src)
		planner = self._setup_fit(src, **settings)
		self.reporter.setup(self, planner, batch_size)

		batch = None
		for batch in self.loop(batch_size, system=system, planner=planner):
			# Note: this runs the optimization step before yielding the batch
			terminate = self.learn(batch)
			yield batch
			if terminate: break

		self._end_fit(batch)
		self.reporter.end(batch)


	def learn(self, batch: Batch) -> bool:
		self._optimizer.step(batch)

		for e in self._events.values():
			e.step(batch)

		reporter = self.reporter
		if reporter is not None:
			reporter.step(batch)

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




