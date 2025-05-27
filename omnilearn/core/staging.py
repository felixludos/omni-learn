from .imports import *
from omniply.gears.mechanics import MechanizedBase, AutoMechanized, MechanizedGame, AbstractMechanics, Mechanics
from omnibelt.staging import AutoStaged as _AutoStaged, Staged as _Staged, AbstractScape



class Scape(Mechanics, AbstractScape):
	pass



class Staged(_Staged, MechanizedBase):
	def _stage(self, scape: AbstractScape):
		stage_mechanics = isinstance(scape, AbstractMechanics) and self._mechanics is None
		if stage_mechanics:
			self.mechanize(scape)

		super()._stage(scape)

		if stage_mechanics:
			self.mechanize(None)



class TopLevelStaged(Staged, AutoMechanized):
	def stage(self, scape: AbstractScape = None):
		if not self.is_staged and scape is None:
			scape = self._auto_mechanics()
		return super().stage(scape)



class AutoStaged(Staged, _AutoStaged):
	pass















