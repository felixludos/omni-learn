from .imports import *
from ..abstract import AbstractPlanner, AbstractDataset, AbstractTrainer, AbstractBatch



class DefaultPlanner(AbstractPlanner):
	def __init__(self, src: AbstractDataset, trainer: AbstractTrainer, *, batch_size: Optional[int] = None,
				 max_steps: Optional[int] = None, max_epochs: Optional[int] = None, step_offset: int = 0, **kwargs):
		if batch_size is None:
			batch_size = src.suggested_batch_size
		super().__init__(**kwargs)
		if max_epochs is not None and src.size is None:
			raise ValueError('max_epochs requires a dataset with a defined size')
		self._src = src
		self._trainer = trainer
		self._batch_size = batch_size
		self._max_steps = max_steps
		self._max_epochs = max_epochs
		self._step_offset = step_offset
		self._step = step_offset

	def to_batch(self, info: JSONDATA, **kwargs) -> AbstractBatch:
		return self._trainer.to_batch(info, **kwargs)

	@property
	def batch_size(self) -> int:
		return self._batch_size

	@property
	def max_steps(self) -> Optional[int]:
		return self._max_steps

	@property
	def max_epochs(self) -> Optional[int]:
		return self._max_epochs

	@property
	def total_steps(self) -> Optional[int]:
		options = []
		if self._max_steps is not None:
			options.append(self._max_steps)
		if self._max_epochs is not None:
			options.append(self._max_epochs * self._src.size // self._batch_size)
		if options:
			return min(options)
		return None

	@property
	def steps_remaining(self) -> Optional[int]:
		expected = self.total_steps
		if expected is not None:
			return max(0, expected - self._step)
		return None

	@property
	def past_steps(self) -> int:
		return self._step

	def __iter__(self) -> Iterator[AbstractBatch]:
		return self

	def __next__(self) -> AbstractBatch:
		if not self.steps_remaining:
			raise StopIteration
		info = self._trainer.generate_batch_info(self._batch_size)
		batch = self.to_batch(info)
		self._step += 1
		return batch



