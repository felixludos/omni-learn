from .imports import *
from ..abstract import AbstractDataset



class AbstractFileDataset(AbstractDataset):
	@property
	def dataroot(self) -> Path:
		raise NotImplementedError



class AbstractEvaluatableDataset(AbstractDataset):
	def as_eval(self) -> AbstractDataset:
		raise NotImplementedError



