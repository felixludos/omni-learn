from .imports import *
from ..core import Batch, ToolKit, Structured
from ..abstract import (AbstractMachine, AbstractTrainer, AbstractModel, AbstractDataset, AbstractOptimizer,
						AbstractEvent, AbstractPlanner, AbstractBatch)
# from omniply.apps.training import TrainerBase as _TrainerBase
from omniply.apps.gaps import Mechanics
from ..mixins import AbstractCheckpointable, AbstractJsonable


@fig.component('default-trainer')
class TrainerBase(fig.Configurable, AutoStaged, AbstractTrainer):
	def __init__(self, dataset: AbstractDataset, model: AbstractModel, optimizer: AbstractOptimizer = None, *,
				 env: Dict[str, AbstractMachine] = None, batch_size: int = None, seed: int = None,
				 name: str = '{dataset.name}_{model.name}_{now.strftime("%y%m%d-%H%M%S")}_{unique[:4]}',
				 max_steps: Optional[int] = None, max_epochs: Optional[int] = None, max_samples: Optional[int] = None,
				 **kwargs):
		if env is None: env = {}
		if seed == 'sample': seed = random.randint(0, 2 ** 31 - 1)
		super().__init__(**kwargs)
		self._name = None
		self._name_format = name
		self._timestamp = datetime.now()
		self._dataset = dataset
		self._model = model
		self._optimizer = optimizer
		self._batch_size = batch_size
		self._seed = seed
		self._env = env
		self._extra_info = {}
		self._max_steps = max_steps
		self._max_samples = max_samples
		self._max_epochs = max_epochs
		self._past_batches = None
		self._past_iterations = None
		self._past_failures = None
		self._plan = None

	@property
	def name(self) -> Optional[str]:
		return self._name

	def remember(self, **info: JSONDATA) -> Self:
		"""
		Remember some information about the training run, eg. hyperparameters, settings, etc.
		This is useful for logging and debugging.
		"""
		self._extra_info.update(info)
		return self

	def train(self):
		self._model.train()

	def eval(self):
		self._model.eval()

	def to(self, device: str = None):
		raise NotImplementedError


	def json(self) -> JSONDATA:
		data = {
			'dataset': self._dataset.json(),
			'model': self._model.json(),
			'optim': self._optimizer.json(),
			'batch_size': self._batch_size,
			'seed': self._dataset.seed if self._dataset is not None else self._seed,
			'extra': self._extra_info,
			'env': {
				key: agent.json() if isinstance(agent, AbstractJsonable) else str(agent)
			for key, agent in self._env.items()},
		}
		return data

	@property
	def dataset(self) -> AbstractDataset:
		return self._dataset

	@property
	def model(self) -> AbstractModel:
		return self._model

	@property
	def optimizer(self) -> AbstractOptimizer:
		return self._optimizer


	def gadgetry(self) -> Iterator[AbstractGadget]:
		yield self._model
		yield self._optimizer
		yield from self._env.values()


	def prepare(self) -> Self:
		# TODO: set name if its not already set
		scape = Mechanics(self.dataset, *self.gadgetry())
		self.stage(scape)
		self.dataset.stage(scape)
		if self._batch_size is None:
			self._batch_size = self.dataset.suggested_batch_size
		for gadget in self.gadgetry():
			if isinstance(gadget, AbstractStaged):
				gadget.stage(scape)

		self._past_iterations = 0
		self._past_batches = 0
		self._past_failures = 0

		if self._name is None:
			self._name = pformat(self._name_format, trainer=self, dataset=self.dataset, model=self.model,
								 optimizer=self.optimizer, now=self._timestamp, unique=urandom(16).hex())

		return self

	def describe(self) -> str: # TODO: should be called from the (first) reporter
		lines = [
			f'Trainer: {self.name}',
			f'Dataset: {self.dataset}',
			f'Model: {self.model}',
			f'Optimizer: {self.optimizer}',
		]
		return '\n'.join(lines) + '\n'

	def pre_loop(self, targets: Iterable[str] = None) -> Optional[JSONDATA]:
		pass

	def post_loop(self, targets: Iterable[str] = None) -> Optional[JSONDATA]:
		pass

	def enumerate(self, num: Optional[int] = None, *, batch_size: Optional[int] = None,
				  gadgets: Iterable[AbstractGadget] = None) -> Iterator[Tuple[int, AbstractBatch]]:
		for batch in self.batches(num, batch_size=batch_size, gadgets=gadgets):
			yield self._past_iterations, batch
	def batches(self, num: Optional[int] = None, *, batch_size: Optional[int] = None, force: bool = False,
				gadgets: Iterable[AbstractGadget] = None) -> Iterator[AbstractBatch]:
		remaining = self.steps_remaining
		if remaining is None:
			todo = num
		elif num is None or force:
			todo = remaining
		else:
			todo = min(remaining, num)
		n = 0
		while todo is None or n < todo:
			yield self.batch(batch_size, gadgets=gadgets)
			n += 1
	_Batch = None
	def batch(self, batch_size: int = None, *extra_gadgets: AbstractGadget,
			  gadgets: Iterable[AbstractGadget] = None, **kwargs) -> AbstractBatch:
		if batch_size is None:
			batch_size = self._batch_size
		info = {'bidx': self._past_batches, 'step': self._past_iterations, 'size': batch_size}
		batch_cls = self._Batch or getattr(self.dataset, '_Batch', None) or Batch
		batch = batch_cls(self, info=info, **kwargs)
		if extra_gadgets:
			batch.extend(extra_gadgets)
		if gadgets is None: # default to the usual gadgets
			batch.include(self.dataset).extend(self.gadgetry())
		else:
			batch.extend(gadgets)
		self._past_batches += 1
		return batch

	@property
	def max_steps(self) -> Optional[int]:
		return self._max_steps
	@property
	def max_epochs(self) -> Optional[int]:
		return self._max_epochs
	@property
	def max_samples(self) -> Optional[int]:
		return self._max_samples
	@property
	def expected_steps(self) -> Optional[int]:
		options = []
		if self._max_steps is not None:
			options.append(self._max_steps)
		if self._max_samples is not None:
			options.append(self._max_samples // self._batch_size)
		if self._max_epochs is not None:
			options.append(self._max_epochs * self.dataset.size // self._batch_size)
		if options:
			return min(options)
		return None
	@property
	def steps_remaining(self) -> Optional[int]:
		expected = self.expected_steps
		if expected is not None:
			return max(0, expected - self.past_steps)
		return None
	@property
	def past_steps(self) -> int:
		return self._past_iterations
	@property
	def past_batches(self) -> int:
		return self._past_batches

	def __iter__(self) -> Iterator[AbstractBatch]:
		return self.batches()
	def __next__(self) -> AbstractBatch:
		return self.batch()

	def learn(self, batch: AbstractBatch) -> bool:
		try:
			self.optimizer.step(batch)
		except:
			self._past_failures += 1
			raise
		self._past_iterations += 1
		return self._terminate_fit(batch)

	def _terminate_fit(self, batch: AbstractBatch) -> bool:
		'''is the training done?'''
		return False



	# def setup(self, src: AbstractDataset, *, device: str = None):
	# 	self._dataset = src
	# 	if device is None:
	# 		device = self._device
	# 	src = src.prepare(device=device)
	# 	system = self._System(src, *self.gadgetry())
	# 	system.mechanize() # sync for gears and spaces
	# 	mech = system.mechanics()
	# 	self.prepare(device=device)
	# 	for e in self._events.values():
	# 		e.setup(self, src, device=device)
	# 	return system


	# def _setup_fit(self, src: AbstractDataset, *, device: str = None, **settings: Any) -> AbstractPlanner:
	# 	return self._planner.setup(src, **settings)


	# def _end_fit(self, batch: Batch) -> None:
	# 	for e in self._events.values():
	# 		e.end(batch)

	#
	# def loop(self, batch_size: int = None, *, system: Structured = None, planner: AbstractPlanner = None):
	# 	if batch_size is None:
	# 		assert self._batch_size is not None, 'batch_size must be provided if not set'
	# 		batch_size = self._batch_size
	# 	if planner is None:
	# 		assert self._dataset is not None, 'planner must be provided if dataset is not set'
	# 		planner = self._planner.setup(self._dataset)
	#
	# 	batch_cls = self._Batch or getattr(self._dataset, '_Batch', None) or Batch
	# 	for info in planner.generate(batch_size):
	# 		batch = batch_cls(info, planner=planner)
	# 		if system is not None:
	# 			batch.include(system)
	# 		yield batch
	#

	# def fit(self, src: AbstractDataset) -> Self:
	# 	'''train the model'''
	# 	for batch in self.fit_loop(src): pass
	# 	return self
	#
	# def fit_loop(self, src: AbstractDataset, *, batch_size: int = None, **settings):
	# 	if batch_size is None:
	# 		batch_size = src.suggest_batch_size() if self._batch_size is None else self._batch_size
	#
	# 	system = self.setup(src)
	# 	planner = self._setup_fit(src, **settings)
	# 	self.reporter.setup(self, planner, batch_size)
	#
	# 	batch = None
	# 	for batch in self.loop(batch_size, system=system, planner=planner):
	# 		# Note: this runs the optimization step before yielding the batch
	# 		terminate = self.learn(batch)
	# 		yield batch
	# 		if terminate: break
	#
	# 	self._end_fit(batch)
	# 	self.reporter.end(batch)
	#
	#
	# def learn(self, batch: Batch) -> bool:
	# 	self._optimizer.step(batch)
	#
	# 	for e in self._events.values():
	# 		e.step(batch)
	#
	# 	reporter = self.reporter
	# 	if reporter is not None:
	# 		reporter.step(batch)
	#
	# 	return self._terminate_fit(batch)



	# def fit_loop(self, src: AbstractDataset, **settings):
	#
	# 	system = self.prepare(src)
	#
	#
	#
	# 	for batch in self.loop(system):
	# 		terminate = self.learn(batch)
	# 		yield batch
	# 		if terminate: break
	#
	#
	#
	# 	pass
	#
	#
	# def fit(self, src: AbstractDataset, **settings) -> Structured:
	# 	for _ in self.fit_loop(src): pass


	# def _stage(self, scape: AbstractMechanics):
	# 	super()._stage(scape)
	# 	self._model.stage(scape)
	# 	self._optimizer.stage(scape)
	# 	for key, e in self._env.items():
	# 		if isinstance(e, AbstractStaged):
	# 			e.stage(scape)
	# 	for key, e in self._events.items():
	# 		if isinstance(e, AbstractStaged):
	# 			e.stage(scape)

	# def _prepare(self, *, device: str = None):
	# 	"""
	# 	Only run once, only uses dataset implicitly - eg through gears. Generally this is not the top level
	#
	# 	use setup() instead
	# 	"""
	# 	self._model.prepare(device=device)
	# 	self._optimizer.setup(self._model)
	# 	self._optimizer.prepare(device=device)
	# 	for e in self._env.values():
	# 		e.prepare(device=device)
	# 	return self


class CheckpointableTrainer(TrainerBase):
	def __init__(self, checkpoint_freq: int = None, **kwargs):
		super().__init__(**kwargs)
		self._my_now = datetime.now().strftime('%Y%m%d_%H%M%S')


	@property
	def name(self) -> str:
		return f'{super().name}_{self._my_now}'


	def checkpoint(self, path: Path) -> None:
		path.mkdir()

		path.joinpath('config.yaml').write_text(str(self._my_config.root))

		self.model.checkpoint(path / 'model')
		self.optimizer.checkpoint(path / 'optimizer')
		if self._dataset is not None:
			self._dataset.checkpoint(path / 'dataset')
		for ident, machine in self._env.items():
			machine.checkpoint(path / ident)

		return path




