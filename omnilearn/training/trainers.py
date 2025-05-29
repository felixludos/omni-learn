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


	def json(self) -> JSONOBJ:
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

	def status(self) -> JSONOBJ:
		pass

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
		self.model.stage(scape)
		self.optimizer.stage(scape)
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

		self.optimizer.add_parameters(self.model.parameters())
		return self


	def describe(self) -> str: # TODO: should be called from the (first) reporter
		rows = [
			('Trainer', self),
			('Dataset', self.dataset),
			('Model', self.model),
			('Optimizer', self.optimizer),
		]
		for name, machine in self._env.items():
			rows.append((f'{name}', machine))

		return tabulate(rows)# + '\n'

	def pre_loop(self, targets: Iterable[str] = None) -> Optional[JSONOBJ]:
		pass

	def post_loop(self, targets: Iterable[str] = None) -> Optional[JSONOBJ]:
		pass

	def summary(self) -> Optional[str]:
		pass


	def enumerate(self, num: Optional[int] = None, *, batch_size: Optional[int] = None,
				  gadgets: Iterable[AbstractGadget] = None) -> Iterator[Tuple[int, AbstractBatch]]:
		for batch in self.iterate(num, batch_size=batch_size, gadgets=gadgets):
			yield self._past_iterations, batch
	def iterate(self, num: Optional[int] = None, *, batch_size: Optional[int] = None, force: bool = False,
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
		return self.iterate()
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



