from .imports import *
from .staging import AutoStaged
from .containers import Structured
from ..abstract import AbstractMachine


# class Machine(Prepared, Structured, AbstractMachine):
class Machine(AutoStaged, Structured, AbstractMachine):
	def _checkpoint_data(self) -> Dict[str, Any]:
		return self.settings()


	def _load_checkpoint_data(self, data: Dict[str, Any], *, unsafe: bool = False) -> None:
		if data != self.settings():
			if unsafe:
				print(f'WARNING: settings do not match: {data} != {self.settings()}')
			else:
				raise ValueError(f'settings do not match: {data} != {self.settings()}')
		if len(data):
			raise NotImplementedError


	def checkpoint(self, path: Path = None) -> Any:
		data = self._checkpoint_data()
		if data is None or not len(data):
			return None
		if path is None:
			return data
		if path.suffix == '': path = path.with_suffix('.pt')
		torch.save(data, path)
		return path


	def load_checkpoint(self, *, path: Path = None, data: Any = None,
						unsafe: bool = False) -> Path | None:
		if data is None:
			assert path is not None, f'must provide path or data (not both)'
			if not path.exists() and path.suffix == '': path = path.with_suffix('.pt')
			assert path.exists(), f'checkpoint file does not exist: {path}'
			data = torch.load(path, map_location='cpu')
		else:
			assert path is None, f'must provide path or data (not both)'
		self._load_checkpoint_data(data, unsafe=unsafe)
		return path


	def settings(self) -> Dict[str, Any]:
		return {}



class AbstractSystem(AbstractStaged):
	@property
	def source(self):
		raise NotImplementedError

	def iterate(self, batch_size: int, **kwargs):
		raise NotImplementedError

	def batch(self, batch_size: int, **kwargs):
		raise NotImplementedError



class System(Machine, AbstractSystem):
	def __init__(self, src: 'AbstractDataset', *other: AbstractGadget, env: Dict[str, AbstractGadget], **kwargs):
		super().__init__(*other, *env.values(), **kwargs)
		self._source = src
		self._env = env


	@property
	def source(self):
		return self._source


	def iterate(self, batch_size: int, **kwargs):
		return self._source.iterate(batch_size, **kwargs)


	def batch(self, batch_size: int, **kwargs):
		return self._source.batch(batch_size, **kwargs)


	def _stage(self, scape: AbstractMechanics):
		super()._stage(scape)
		for key, e in self._env.items():
			if isinstance(e, AbstractStaged):
				e.stage(scape)
		for key, e in self._events.items():
			if isinstance(e, AbstractStaged):
				e.stage(scape)
















