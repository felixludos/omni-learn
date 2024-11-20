from .imports import *



class AbstractSpace:
	@property
	def batched(self) -> bool:
		raise NotImplementedError

	@property
	def size(self) -> int:
		raise NotImplementedError

	@property
	def lower_bound(self) -> Optional[float]:
		raise NotImplementedError

	@property
	def upper_bound(self) -> Optional[float]:
		raise NotImplementedError

	@property
	def bounds(self) -> tuple[Optional[float], Optional[float]]:
		return self.lower_bound, self.upper_bound

	@property
	def dtype(self) -> 'torch.dtype':
		raise NotImplementedError

	def shape(self, batch_size: Optional[int] = None) -> tuple[int]:
		raise NotImplementedError

	def json(self) -> dict:
		raise NotImplementedError