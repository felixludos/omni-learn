from .imports import *
from .abstract import AbstractSpace


class SpaceBase(AbstractSpace):
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
	def dtype(self) -> 'torch.dtype':
		return torch.float32


class Scalar(SpaceBase):
	def shape(self, batch_size: Optional[int] = None) -> tuple:
		return ()


class Categorical(SpaceBase):
	def __init__(self, n: int):
		self._n = n

	def size(self) -> int:
		return self._n

	def shape(self, batch_size: Optional[int] = None) -> tuple:
		return (batch_size, 1)

	def dtype(self) -> 'torch.dtype':
		return torch.int64


class Vector(SpaceBase):
	def __init__(self, dim: int):
		self._dim = dim

	def shape(self, batch_size: Optional[int] = None) -> tuple:
		return (batch_size, self._dim)


class Boolean(Vector):
	@property
	def dtype(self) -> 'torch.dtype':
		return torch.bool


class Spatial(SpaceBase):
	def __init__(self, C: int, *, spatial: tuple, channel_first: bool = True):
		self._channels = C
		self._spatial = spatial
		self._channel_first = channel_first

	@property
	def channels(self) -> int:
		return self._channels

	def shape(self, batch_size: Optional[int] = None) -> tuple:
		return (batch_size, self._channels, *self._spatial) if self._channel_first \
			else (batch_size, *self._spatial, self._channels)


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
	def dtype(self) -> 'torch.dtype':
		return torch.uint8 if self._as_bytes else torch.float32

	@property
	def lower_bound(self) -> Optional[Union[int, float]]:
		return 0 if self._as_bytes else 0.

	@property
	def upper_bound(self) -> Optional[Union[int, float]]:
		return 255 if self._as_bytes else 1.


class Volume(Spatial):
	def __init__(self, C: int, D: int, H: int, W: int, **kwargs):
		super().__init__(C=C, spatial=(D, H, W), **kwargs)

	@property
	def depth(self) -> int:
		return self._spatial[0]

	@property
	def height(self) -> int:
		return self._spatial[1]

	@property
	def width(self) -> int:
		return self._spatial[2]

