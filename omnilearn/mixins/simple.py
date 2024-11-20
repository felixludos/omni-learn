from .imports import *
from .abstract import AbstractPrepared



class Prepared(AbstractPrepared):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._is_prepared = False


	def prepare(self, **kwargs) -> Self:
		if not self._is_prepared:
			self._prepare(**kwargs)
			self._is_prepared = True
		return self


	def _prepare(self, *, device: Optional[str] = None):
		pass






