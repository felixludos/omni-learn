from .imports import *
from .common import Machine

from ..data import FileDatasetBase



class Dataset(Machine, FileDatasetBase):
	def _prepare(self, **kwargs):
		self.load(**kwargs)
		return self


