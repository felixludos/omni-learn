from .imports import *
from ..spaces.abstract import AbstractSpace
from ..abstract import AbstractTrainer, AbstractBatch
from ..mixins import AbstractBatchable
from omniply.apps import DictGadget



class Context(omniply_Context):
	pass



class ToolKit(omniply_ToolKit):
	_space_of_default = object()
	def space_of(self, gizmo: str, default: Any = _space_of_default) -> AbstractSpace:
		try:
			return self.mechanics().grab(self.gap(gizmo))
		except GrabError:
			if default is self._space_of_default:
				raise
			return default



class Structured(omniply_Structured):
	pass



class Mechanism(omniply_Mechanism):
	pass


class BatchInfo(DictGadget):
	pass


class NoNewBatches(Exception):
	pass


class Batch(Context, AbstractBatch):
	_BatchInfo = BatchInfo
	def __init__(self, source: AbstractBatchable, info: Dict[str, Any] = None, *,
				 allow_draw: bool = True, **kwargs):
		if isinstance(info, dict):
			info = self._BatchInfo(info)
		super().__init__(**kwargs)
		self._info = info
		self._source = source
		self._allow_draw = allow_draw
		self.include(info)

	def replace(self, new_info: Dict[str, Any]) -> Self:
		# TODO: track attributes set during use and remove those when replacing
		self.clear_cache()
		# TODO: potentially verify that new_info has the same keys as the original
		self._info.data.clear()
		self._info.data.update(new_info)
		return self

	def new(self, size: int = None) -> 'Batch':
		if self._allow_draw:
			return self.source.batch(size, self.gadgetry())
		raise NoNewBatches(f'creating new batches using the current batch is currently not allowed')

	def gadgetry(self) -> Iterator[AbstractGadget]:
		for gadget in self.vendors():
			if gadget is not self._info:
				yield gadget

	@property
	def size(self) -> int:
		return self.grab('size')




