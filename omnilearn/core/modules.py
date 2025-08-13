from typing import Mapping

from omniply.gems.abstract import AbstractGeode
from .imports import *
from .staging import AutoStaged
from .containers import Structured
from ..mixins import AbstractCheckpointable, AbstractNamed
from ..abstract import AbstractMachine, AbstractEvent, AbstractTrainer, AbstractDataset, AbstractBatch


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


	def _stage(self, scape: AbstractScape):
		self._stage_geodes(scape)
		return super()._stage(scape)


	# def settings(self) -> Dict[str, Any]:
	# 	return {}



class AbstractSystem(AbstractNamed, AbstractStaged, AbstractCheckpointable):
	@property
	def source(self):
		raise NotImplementedError

	def iterate(self, batch_size: int, **kwargs):
		raise NotImplementedError

	def batch(self, batch_size: int, **kwargs):
		raise NotImplementedError



# class SystemBase(Machine, AbstractSystem):
# 	def __init__(self, src: 'AbstractDataset', *args, **kwargs):
# 		super().__init__(*args, **kwargs)
# 		self._source = src
#
# 	@property
# 	def source(self):
# 		return self._source
#
# 	def gadgetry(self) -> Iterator[AbstractGadget]:
# 		yield from self._env.values()
#
# 	def iterate(self, batch_size: int = None, **kwargs):
# 		return self._source.iterate(batch_size, **kwargs)
#
# 	def enumerate(self, batch_size: int = None, **kwargs):
# 		pass
#
# 	def batch(self, batch_size: int = None, **kwargs):
# 		return self._source.batch(batch_size, **kwargs)
#
# 	def _stage(self, scape: AbstractMechanics = None):
# 		if scape is None:
# 			# create scape based on content
# 			scape = Mechanics(self.source, *self.gadgetry())
# 		super()._stage(scape)
# 		self.source.stage(scape)
# 		for gadget in self.gadgetry():
# 			if isinstance(gadget, AbstractStaged):
# 				gadget.stage(scape)
# 		return scape



# class PlannedSystem(SystemBase):
# 	def __init__(self, src: 'AbstractDataset', trainer: AbstractTrainer, **kwargs):
# 		super().__init__(src, **kwargs)
# 		self._trainer = trainer
#
# 	@property
# 	def name(self):
# 		return self._trainer.name_using(self.source)
#
# 	def gadgetry(self) -> Iterator[AbstractGadget]:
# 		yield from self._trainer.gadgetry()
#
#
#
# class System(SystemBase):
# 	def __init__(self, src: 'AbstractDataset', *other, env: Dict[str, AbstractGadget], **kwargs):
# 		super().__init__(src, *other, *env.values(), **kwargs)
# 		self._env = env
# 		self._other = other
#
# 	def gadgetry(self) -> Iterator[AbstractGadget]:
# 		yield from self._env.values()



class Event(AbstractEvent):
	def setup(self, trainer: AbstractTrainer, src: AbstractDataset, *, device: Optional[str] = None) -> Self:
		return self

	def step(self, batch: AbstractBatch) -> None:
		pass

	def end(self, last_batch: AbstractBatch = None) -> None:
		pass
















