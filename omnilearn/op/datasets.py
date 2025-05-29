from .imports import *
from .common import Machine

from ..data import FileDatasetBase, DefaultSelector



class Dataset(Machine, FileDatasetBase):
	def setup(self):
		self.load()
		self.selector = DefaultSelector(self, )
		return self

	def json(self):
		raise NotImplementedError('include selector hyperparameters')

	def status(self):
		raise NotImplementedError('include selector status')

	@tool('indices')
	def select_indices(self, size: int) -> np.ndarray:
		info = self.selector.draw(size)
		inds = info['index']
		return inds


