from .imports import *
from .mixins import *
from .spaces import AbstractSpace



class AbstractSource(AbstractNamed, AbstractSized):
	def iterate(self, batch_size: int, **kwargs) -> Iterator['AbstractBatch']:
		raise NotImplementedError


	def batch(self, batch_size: int, **kwargs) -> 'AbstractBatch':
		raise NotImplementedError


	def actualize(self, info: Dict[str, Any], planner: 'AbstractSelector' = None) -> 'AbstractBatch':
		raise NotImplementedError


	def enumerate(self, batch_size: int, **kwargs) -> Iterator[Tuple[int, 'AbstractBatch']]:
		raise NotImplementedError



class AbstractSystem(AbstractPlanning, AbstractSource):
	@property
	def source(self) -> AbstractSource:
		raise NotImplementedError



class AbstractSelector(AbstractJsonable):#(AbstractPlanning, AbstractSource):
	def draw(self, size: int) -> Dict[str, Any]:
		'''create the info for a new batch'''
		raise NotImplementedError


	def generate(self, step_size: int) -> Iterator[Dict[str, Any]]:
		'''generate batch infos with given step size'''
		raise NotImplementedError



class AbstractMachine(AbstractCheckpointable, AbstractSettings, AbstractStaged, AbstractGadget):
	pass



class AbstractEvent(AbstractMachine):
	def prepare(self):
		pass


	def step(self, batch: 'AbstractBatch') -> None:
		pass


	def end(self, last_batch: Optional['AbstractBatch'] = None) -> None:
		pass



class AbstractDataset(AbstractSource, AbstractMachine):
	@property
	def suggested_batch_size(self) -> int:
		raise NotImplementedError



class AbstractBatch(AbstractDataset, AbstractGame): # AbstractIndustry
	def new(self, size: int = None) -> 'AbstractBatch':
		raise NotImplementedError


	@property
	def plan(self) -> Optional[AbstractSelector]:
		raise NotImplementedError



# class AbstractEngine(AbstractSettings): # AbstractIndustry
# 	def loop(self) -> Iterator[AbstractGame]:
# 		raise NotImplementedError
#
#
# 	def run(self) -> JSONDATA:
# 		for _ in self.loop(): pass



class AbstractModel(AbstractNamed, AbstractMachine):
	pass



class AbstractOptimizer(AbstractNamed):
	def step(self, batch: AbstractBatch) -> None:
		raise NotImplementedError


	@property
	def objective(self) -> str:
		'''key of the overall objective being optimized (default should be "loss")'''
		raise NotImplementedError


	@property
	def objective_direction(self) -> int:
		'''direction of the objective (1 for maximization, -1 for minimization)'''
		raise NotImplementedError



class AbstractTrainer(AbstractNamed, AbstractCheckpointable, AbstractBatchable, AbstractJsonable):
	@property
	def dataset(self) -> AbstractDataset:
		raise NotImplementedError
	@property
	def model(self) -> AbstractModel:
		raise NotImplementedError
	@property
	def optimizer(self) -> AbstractOptimizer:
		raise NotImplementedError

	def train(self):
		raise NotImplementedError
	def eval(self):
		raise NotImplementedError
	def to(self, device: str = None):
		raise NotImplementedError

	def gadgetry(self) -> Iterator[AbstractGadget]:
		raise NotImplementedError

	def prepare(self) -> Self:
		raise NotImplementedError

	def remember(self, **info: JSONDATA) -> Self:
		raise NotImplementedError

	def describe(self) -> str:
		raise NotImplementedError

	def learn(self, batch: AbstractBatch) -> bool:
		raise NotImplementedError

	def _terminate_fit(self, batch: AbstractBatch) -> bool:
		raise NotImplementedError

	def status(self) -> JSONOBJ:
		raise NotImplementedError

	def summary(self) -> str:
		raise NotImplementedError


class AbstractPlanner:
	def to_batch(self, info: Dict[str, Any]) -> 'AbstractBatch':
		raise NotImplementedError

	@property
	def batch_size(self) -> int:
		'''default batch size for the planner'''
		raise NotImplementedError

	@property
	def max_steps(self) -> Optional[int]:
		'''maximum number of steps to take in a plan'''
		raise NotImplementedError

	@property
	def max_epochs(self) -> Optional[int]:
		'''maximum number of epochs to take in a plan'''
		raise NotImplementedError

	@property
	def total_steps(self) -> Optional[int]:
		raise NotImplementedError

	@property
	def steps_remaining(self) -> Optional[int]:
		raise NotImplementedError

	@property
	def past_steps(self) -> Optional[int]:
		raise NotImplementedError

	def __iter__(self):
		raise NotImplementedError

	def __next__(self):
		raise NotImplementedError


# class AbstractReporter(AbstractGadget):
# 	def setup(self, trainer: AbstractTrainer, planner: AbstractPlanner, batch_size: int) -> Self:
# 		return self
#
#
# 	def step(self, batch: AbstractBatch) -> None:
# 		raise NotImplementedError
#
#
# 	def end(self, last_batch: AbstractBatch = None) -> None:
# 		raise NotImplementedError
#
#
# 	def checkpointed(self, path: str) -> None:
# 		raise NotImplementedError

