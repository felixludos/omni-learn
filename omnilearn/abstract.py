from .imports import *
from .mixins import *
from .spaces import AbstractSpace

from omniply.apps.training.abstract import AbstractDataset as AbstractDatasetBase, AbstractPlanner as AbstractPlannerBase, AbstractBatch, AbstractTrainer as AbstractTrainerBase



class AbstractPlanner(AbstractPlannerBase):
	def budget(self, **settings):
		raise NotImplementedError


	def expected_samples(self, step_size: int) -> Optional[int]:
		raise NotImplementedError



class AbstractMachine(AbstractPrepared, AbstractCheckpointable, AbstractSettings, AbstractGadget):
	def prepare(self, *, device: str = None) -> Self:
		raise NotImplementedError



class AbstractDataset(AbstractNamed, AbstractDatasetBase, AbstractMachine):
	def load(self, *, device: Optional[str] = None) -> Self:
		raise NotImplementedError


	def suggest_batch_size(self) -> int:
		pass



class AbstractEvaluatableDataset(AbstractDataset):
	def as_eval(self) -> AbstractDataset:
		raise NotImplementedError



class AbstractModel(AbstractNamed, AbstractMachine):
	pass



class AbstractOptimizer(AbstractNamed, AbstractMachine):
	def setup(self, model: AbstractModel) -> Self:
		raise NotImplementedError


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



class AbstractTrainer(AbstractNamed, AbstractTrainerBase):
	@property
	def settings(self) -> Dict[str, Any]:
		raise NotImplementedError


	@property
	def model(self) -> AbstractModel:
		raise NotImplementedError


	@property
	def optimizer(self) -> AbstractOptimizer:
		raise NotImplementedError


	@property
	def dataset(self) -> AbstractDataset:
		raise NotImplementedError



class AbstractReporter(AbstractGadget):
	def setup(self, trainer: AbstractTrainer, planner: AbstractPlanner, batch_size: int) -> Self:
		return self


	def step(self, batch: AbstractBatch) -> None:
		raise NotImplementedError


	def end(self, last_batch: AbstractBatch = None) -> None:
		raise NotImplementedError


	def checkpointed(self, path: str) -> None:
		raise NotImplementedError



class AbstractEvent(AbstractMachine):
	def setup(self, trainer: AbstractTrainer, src: AbstractDataset, *, device: Optional[str] = None) -> Self:
		raise NotImplementedError


	def step(self, batch: AbstractBatch) -> None:
		raise NotImplementedError


	def end(self, last_batch: AbstractBatch = None) -> None:
		raise NotImplementedError


