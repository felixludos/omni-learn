from .imports import *
from .mixins import *
from .spaces import AbstractSpace



class AbstractSource(AbstractNamed, AbstractSized):
	def iterate(self, batch_size: int, **kwargs) -> Iterator['AbstractBatch']:
		raise NotImplementedError


	def batch(self, batch_size: int, **kwargs) -> 'AbstractBatch':
		raise NotImplementedError


	def actualize(self, info: Dict[str, Any], planner: 'AbstractPlanner' = None) -> 'AbstractBatch':
		raise NotImplementedError



class AbstractSystem(AbstractPlanning, AbstractSource):
	@property
	def source(self) -> AbstractSource:
		raise NotImplementedError



class AbstractPlanner(AbstractPlanning, AbstractSource):
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



class AbstractBatch(AbstractDataset, AbstractIndustry, AbstractGame):
	def new(self, size: int = None) -> 'AbstractBatch':
		raise NotImplementedError


	@property
	def plan(self) -> Optional[AbstractPlanner]:
		raise NotImplementedError



class AbstractEngine(AbstractSettings, AbstractIndustry):
	def loop(self) -> Iterator[AbstractGame]:
		raise NotImplementedError


	def run(self) -> JSONABLE:
		for _ in self.loop(): pass



class AbstractModel(AbstractNamed, AbstractMachine):
	pass



class AbstractOptimizer(AbstractNamed, AbstractEvent):
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



class AbstractTrainer(AbstractNamed, AbstractEngine):
	@property
	def model(self) -> AbstractModel:
		raise NotImplementedError


	@property
	def dataset(self) -> AbstractDataset:
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

