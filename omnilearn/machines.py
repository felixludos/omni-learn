from .imports import *
from .abstract import AbstractMachine, AbstractEvent, AbstractTrainer, AbstractDataset, AbstractBatch
from .core import Structured
from .mixins import Prepared



class Machine(Prepared, Structured, AbstractMachine):
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



class Event(Machine, AbstractEvent):
	def setup(self, trainer: AbstractTrainer, src: AbstractDataset, *, device: Optional[str] = None) -> Self:
		return self

	def step(self, batch: AbstractBatch) -> None:
		pass

	def end(self, last_batch: AbstractBatch = None) -> None:
		pass





