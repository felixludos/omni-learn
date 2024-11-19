from .imports import *
from .abstract import AbstractSpace


class SimpleSpace(AbstractSpace):
	@property
	def batched(self) -> bool:
		return self.shape()[0] is None

	@property
	def size(self) -> int:
		return int(np.prod([d for d in self.shape() if d is not None]).item())

	@property
	def lower_bound(self) -> Optional[float]:
		return None

	@property
	def upper_bound(self) -> Optional[float]:
		return None

	@property
	def dtype(self) -> torch.dtype:
		return torch.float32


class Scalar(SimpleSpace):
	def shape(self, batch_size: Optional[int] = None) -> tuple[int]:
		return ()


class Vector(SimpleSpace):
	def __init__(self, dim: int):
		self._dim = dim

	def shape(self, batch_size: Optional[int] = None) -> tuple[int]:
		return (batch_size, self._dim)


class Boolean(Vector):
	def __init__(self, dim: int = 1):
		super().__init__(dim=dim)

	@property
	def dtype(self) -> torch.dtype:
		return torch.bool


class Spatial(SimpleSpace):
	def __init__(self, C: int, *, spatial: tuple, channel_first: bool = True):
		self._C = C
		self._spatial = spatial
		self._channel_first = channel_first

	def shape(self, batch_size: Optional[int] = None) -> tuple[int]:
		return (batch_size, self._C, *self._spatial) if self._channel_first \
			else (batch_size, *self._spatial, self._C)


class Sequence(Spatial):
	def __init__(self, C: int, L: int, **kwargs):
		super().__init__(C=C, spatial=(L,), **kwargs)

	@property
	def length(self) -> int:
		return self._spatial[0]


class Image(Spatial):
	def __init__(self, C: int, H: int, W: int, **kwargs):
		super().__init__(C=C, spatial=(H, W), **kwargs)

	@property
	def height(self) -> int:
		return self._spatial[0]

	@property
	def width(self) -> int:
		return self._spatial[1]


class Pixels(Image):
	def __init__(self, C: int, H: int, W: int, *, as_bytes: bool = False, **kwargs):
		super().__init__(C=C, H=H, W=W, **kwargs)
		self._as_bytes = as_bytes

	@property
	def dtype(self) -> torch.dtype:
		return torch.uint8 if self._as_bytes else torch.float32

	@property
	def lower_bound(self) -> Optional[Union[int, float]]:
		return 0 if self._as_bytes else 0.

	@property
	def upper_bound(self) -> Optional[Union[int, float]]:
		return 255 if self._as_bytes else 1.

