from .imports import *
from .abstract import AbstractPrepared, AbstractStaged



class Prepared(AbstractPrepared):
	_is_prepared: bool = False


	def prepare(self, **kwargs) -> Self:
		if not self._is_prepared:
			self._prepare(**kwargs)
			self._is_prepared = True
		return self


	def _prepare(self, *, device: Optional[str] = None):
		pass



class Staged(AbstractStaged):
	def stage(self, stage: dict[str, Any]):
		raise NotImplementedError




