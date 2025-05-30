from .imports import *
from .common import Machine

from ..data import FileDatasetBase, InfiniteSelector



class Dataset(Machine, FileDatasetBase):
	_Selector = InfiniteSelector
	def __init__(self, *, selector: AbstractSelector = None, **kwargs):
		if selector is None:
			selector = self._Selector()
		super().__init__(**kwargs)
		self._selector = selector

	def setup(self, *, device: Optional[str] = None):
		self._selector.reset(self.size)

	def json(self):
		raise NotImplementedError('include selector hyperparameters')

	def status(self):
		raise NotImplementedError('include selector status')

	@tool('indices')
	def select_indices(self, size: int) -> np.ndarray:
		inds = self._selector.draw(size)
		return inds


