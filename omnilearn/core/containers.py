from .imports import *
from ..spaces.abstract import AbstractSpace



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



class Batch(omniply_Batch):
	pass




