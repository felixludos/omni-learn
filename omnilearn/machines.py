from .imports import *
from .abstract import AbstractMachine, AbstractEvent, AbstractTrainer, AbstractDataset, AbstractBatch
from .core import Structured
from .mixins import Prepared




class Event(Machine, AbstractEvent):
	def setup(self, trainer: AbstractTrainer, src: AbstractDataset, *, device: Optional[str] = None) -> Self:
		return self

	def step(self, batch: AbstractBatch) -> None:
		pass

	def end(self, last_batch: AbstractBatch = None) -> None:
		pass





