from .imports import *
from .abstract import AbstractPrepared



class Prepared(AbstractPrepared):
	_is_prepared: bool = False


	def prepare(self, **kwargs) -> Self:
		if not self._is_prepared:
			self._prepare(**kwargs)
			self._is_prepared = True
		return self


	def _prepare(self, *, device: Optional[str] = None):
		pass






