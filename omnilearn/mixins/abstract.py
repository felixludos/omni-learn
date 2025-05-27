from .imports import *
from omnibelt.staging import AbstractStaged
from omnibelt import JSONABLE



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



class AbstractCheckpointable:
	def checkpoint(self, path: Path = None):
		raise NotImplementedError


	def load_checkpoint(self, *, path: Path = None, data: Any = None):
		raise NotImplementedError



class AbstractSettings:
	def settings(self) -> Dict[str, JSONABLE]:
		raise NotImplementedError



class AbstractIndustry:
	def gadgetry(self) -> Iterator[AbstractGadget]:
		raise NotImplementedError



