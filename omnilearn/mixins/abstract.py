from .imports import *
from omnibelt.staging import AbstractStaged
from omnibelt import JSONDATA, Jsonable as AbstractJsonable


class AbstractNamed:
	@property
	def name(self) -> str:
		raise NotImplementedError


class AbstractSized:
	@property
	def size(self) -> Optional[int]:
		raise NotImplementedError



class AbstractPlanning:
	def expected_iterations(self, step_size: int) -> Optional[int]:
		raise NotImplementedError


	def expected_samples(self, step_size: int) -> Optional[int]:
		raise NotImplementedError


class AbstractBatchable:
	def enumerate(self, num: Optional[int] = None, *, batch_size: Optional[int] = None,
				  gadgets: Iterable[AbstractGadget] = None) -> Iterator[Tuple[int, 'AbstractBatch']]:
		raise NotImplementedError
	def iterate(self, num: Optional[int] = None, *, batch_size: Optional[int] = None, force: bool = False,
				gadgets: Iterable[AbstractGadget] = None) -> Iterator['AbstractBatch']:
		raise NotImplementedError
	def batch(self, batch_size: int = None, *extra_gadgets: AbstractGadget,
			  gadgets: Iterable[AbstractGadget] = None) -> 'AbstractBatch':
		raise NotImplementedError


class AbstractCheckpointable:
	def checkpoint(self, path: Path = None):
		raise NotImplementedError


	def load_checkpoint(self, *, path: Path = None, data: Any = None):
		raise NotImplementedError



class AbstractSettings:
	pass
	# def settings(self) -> Dict[str, JSONDATA]:
	# 	raise NotImplementedError



class AbstractIndustry:
	def gadgetry(self) -> Iterator[AbstractGadget]:
		raise NotImplementedError


class AbstractData(AbstractGadget):  # TODO: make fingerprinted
	# def copy(self):
	# 	return duplicate_instance(self) # shallow copy

	def _title(self):
		return getattr(self, 'name', self.__class__.__name__)

	@property
	def title(self):
		return self._title()

	def __repr__(self):
		base = super().__repr__()
		title = self.title
		if '(' in base:
			return f'{title}({base.split("(", 1)[1]}'
		return base


class AbstractCountableData(AbstractData):
	def _title(self):
		return super()._title() if self.size is None else f'{super()._title()}[{self.size}]'

	@property
	def size(self):
		raise NotImplementedError



