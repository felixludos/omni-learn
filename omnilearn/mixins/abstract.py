from .imports import *

class AbstractNamed:
	@property
	def name(self) -> str:
		raise NotImplementedError


class AbstractPrepared:
	def prepare(self) -> Self:
		raise NotImplementedError



class AbstractStaged:
	def stage(self, stage: dict[str, Any]):
		raise NotImplementedError



class AbstractCheckpointable:
	def checkpoint(self, path: Path = None):
		raise NotImplementedError


	def load_checkpoint(self, *, path: Path = None, data: Any = None):
		raise NotImplementedError



class AbstractSettings:
	def settings(self) -> Dict[str, Any]:
		raise NotImplementedError


