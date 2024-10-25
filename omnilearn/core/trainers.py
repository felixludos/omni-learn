from .imports import *
from .abstract import AbstractModel, AbstractDataset, AbstractOptimizer, AbstractTrainer, AbstractEvent, AbstractPlanner, AbstractReporter
from omniply.apps.training import DynamicTrainerBase as _DynamicTrainerBase



class TrainerBase(_DynamicTrainerBase):
	_Reporter = None # TODO
	def __init__(self, model: AbstractModel, optimizer: AbstractOptimizer, *, reporter: AbstractEvent = None, device: str = None, **kwargs):
		if reporter is None:
			reporter = self._Reporter()
		super().__init__(**kwargs)
		self._model = model
		self._optimizer = optimizer
		self._reporter = reporter
		self._device = device
	

	def gadgetry(self) -> Iterator[AbstractGadget]:
		yield self._model
		yield self._optimizer
		yield super().gadgetry()


	def _setup_fit(self, src: AbstractDataset, *, device: str = None, **settings: Any) -> AbstractPlanner:
		if device is None:
			device = self._device
		
		planner = self._planner.setup(src, **settings)

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




