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

	def __str__(self) -> str:
		return f'{" x ".join("_" if d is None else str(d) for d in self.shape())}'

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}({", ".join("_" if d is None else str(d) for d in self.shape())})'

	def json(self) -> dict:
		return {'type': self.__class__.__name__}



class Tensor(SpaceBase):
	def __init__(self, *shape: int, batched: bool = True, dtype: 'torch.dtype' = None, **kwargs):
		dtype = dtype or torch.float32
		super().__init__(**kwargs)
		self._shape = shape
		self._dtype = dtype
		self._batched = batched


	def shape(self, batch_size: Optional[int] = None) -> tuple:
		return (batch_size, *self._shape) if self.batched else self._shape


	def json(self) -> dict:
		return {**super().json(), 'shape': list(self._shape), 'batched': self._batched, 'dtype': str(self._dtype)}


	@property
	def batched(self) -> bool:
		return self._batched


	@property
	def dtype(self) -> 'torch.dtype':
		return self._dtype



class Scalar(Tensor):
	def __init__(self, *, dtype: 'torch.dtype' = torch.float32, batched: bool = False):
		super().__init__(dtype=dtype, batched=batched)


	def json(self):
		return {'type': self.__class__.__name__}



class Vector(Tensor):
	def __init__(self, dim: int, *, dtype: 'torch.dtype' = None, batched: bool = True):
		dtype = dtype or torch.float32
		super().__init__(dim, dtype=dtype, batched=batched)



class Logits(Vector):
	pass



class Bounded(Vector):
	def __init__(self, *args, lower: float = None, upper: float = None, **kwargs):
		super().__init__(*args, **kwargs)
		self._lower = lower
		self._upper = upper

	@property
	def lower_bound(self) -> Optional[float]:
		return self._lower

	@property
	def upper_bound(self) -> Optional[float]:
		return self._upper

	def json(self) -> dict:
		return {**super().json(), 'lower': self._lower, 'upper': self._upper}



class Boolean(Tensor):
	def __init__(self, *args, dtype: 'torch.dtype' = None, **kwargs):
		dtype = dtype or torch.bool
		super().__init__(*args, dtype=dtype, **kwargs)



class Categorical(SpaceBase):
	def __init__(self, n: Union[Iterable[int], Iterable[str], int]):
		if isinstance(n, int):
			n = range(n)
		self._classes = tuple(map(str, n))

	@property
	def n(self) -> int:
		return len(self._classes)
	
	@property
	def class_names(self) -> tuple:
		return self._classes

	@property
	def size(self) -> int:
		return self.n

	def shape(self, batch_size: Optional[int] = None) -> tuple:
		return (batch_size, 1)

	def dtype(self) -> 'torch.dtype':
		return torch.int

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}({self.n})'

	def json(self) -> dict:
		return {'type': 'categorical', 'classes': self._classes}



class BooleanCategorical(Categorical):
	def shape(self, batch_size = None):
		return (batch_size, self.n)
	
	def dtype(self):
		return torch.bool



class Spatial(SpaceBase):
	def __init__(self, C: int, *, spatial: tuple, channels_first: bool = True):
		self._channels = C
		self._spatial = spatial
		self._channels_first = channels_first

	@property
	def channels(self) -> int:
		return self._channels

	def shape(self, batch_size: Optional[int] = None) -> tuple:
		return (batch_size, self._channels, *self._spatial) if self._channels_first \
			else (batch_size, *self._spatial, self._channels)

	def json(self) -> dict:
		return {'type': 'spatial', 'channels': self._channels, 'spatial': list(self._spatial),
				'channels_first': self._channels_first}


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

	def json(self) -> dict:
		data = super().json()
		return {**data, 'as_bytes': self._as_bytes}


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

