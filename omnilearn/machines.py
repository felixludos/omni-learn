from .imports import *
from .abstract import AbstractMachine, AbstractEvent
from .core import ToolKit



class Machine(ToolKit, AbstractMachine):
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
			if path.suffix == '': path = path.with_suffix('.pt')
			assert path.exists(), f'checkpoint file does not exist: {path}'
			data = torch.load(path)
		else:
			assert path is None, f'must provide path or data (not both)'
		self._load_checkpoint_data(data, unsafe=unsafe)
		return path


	def settings(self) -> Dict[str, Any]:
		return {}


	# def indicators(self) -> Iterator[str]:
	# 	yield from ()


class Event(Machine, AbstractEvent):
	pass






