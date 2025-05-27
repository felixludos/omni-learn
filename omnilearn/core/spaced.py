from ..imports import *
from .imports import *
from ..spaces import Scalar, Vector, Pixels
from ..spaces.abstract import AbstractSpace

from omniply.apps.gaps import StaticGearCraft
from omniply.core.gadgets import SingleGadgetBase
from omniply.core.tools import AbstractCrafty, CraftBase, ToolCraftBase, ToolDecoratorBase
from omniply.gears.gears import GearCraftBase, GearSkill



class SpaceSkill(SingleGadgetBase):
	def __init__(self, *, parse: bool = True, **kwargs):
		super().__init__(**kwargs)
		self._allow_parse = parse


	def _grab_from(self, ctx: 'AbstractGame') -> Any:
		raw = super()._grab_from(ctx)
		return self._parse_raw_space(raw) if self._allow_parse else raw


	def _parse_raw_space(self, raw: Any) -> AbstractSpace:
		if isinstance(raw, AbstractSpace):
			return raw

		if isinstance(raw, int):
			return Vector(raw)
		elif isinstance(raw, tuple):
			if len(raw) == 2:
				print(f'WARNING: 2D space interpretted as a grayscale uint image: {raw}')
				return Pixels(1, *raw, as_bytes=True)
			elif len(raw) == 3:
				d1, d2, d3 = raw
				if d1 in {1, 3} and d3 not in {1, 3}:
					print(f'WARNING: 3D space interpretted as an float image with {d1} channels: {raw}')
					return Pixels(*raw, channel_first=True, as_bytes=False)
				elif d3 in {1, 3} and d1 not in {1, 3}:
					print(f'WARNING: 3D space interpretted as an float image with {d3} channels: {raw}')
					return Pixels(d3, d1, d2, channel_first=False, as_bytes=False)

		raise ValueError(f'Cannot interpret space from tuple: {raw!r}')


class SpaceCraft(GearCraftBase):
	def __init__(self, gizmo: str, *, parse: bool = False, **kwargs):
		super().__init__(gizmo=gizmo, **kwargs)
		self._parse = parse


	def as_skill(self, owner: AbstractCrafty, *, parse: bool = None, **kwargs) -> GearSkill:
		if parse is None:
			parse = self._parse
		return super().as_skill(owner, parse=parse, **kwargs)




class StaticSpace(SpaceCraft, StaticGearCraft):
	class _GearSkill(SpaceSkill, StaticGearCraft._GearSkill):
		pass



class ShortcutToolCraft(ToolCraftBase):
	_Space: Type[SpaceCraft] = None
	def space(self, fn: Callable) -> SpaceCraft: # for use as a decorator (much like `property.setter`)
		# assert self._space is self._space_empty_value, f'Space is already set: {self._space}'
		if not isinstance(self._gizmo, str) and len(self._gizmo) > 1:
			raise ValueError('Cannot create a space from a tool that produces multiple gizmos')
		return self._Space(self._gizmo, fn=fn)


	_Gear: Type[omniply_gear] = None
	def gear(self, fn: Callable) -> omniply_gear: # for use as a decorator (much like `property.setter`)
		if not isinstance(self._gizmo, str) and len(self._gizmo) > 1:
			raise ValueError('Cannot create a space from a tool that produces multiple gizmos')
		return self._Gear(self._gizmo, fn=fn)



class SpacedToolCraft(ToolCraftBase):
	_space_empty_value = object()
	def __init__(self, *gizmos: str, space: Any = _space_empty_value, parse_space: bool = True, **kwargs):
		super().__init__(*gizmos, **kwargs)
		self._space = space
		self._parse_space = parse_space


	def emit_craft_items(self, owner: 'AbstractCrafty') -> Iterator['AbstractCraft']:
		yield from super().emit_craft_items(owner)
		spacecraft = self._auto_space_craft()
		if spacecraft is not None:
			yield spacecraft

	_StaticSpace = StaticSpace
	def _auto_space_craft(self, gizmo: str = None, value: Any = _space_empty_value, parse: bool = None) -> StaticSpace:
		if value is self._space_empty_value:
			value = self._space
		if value is not self._space_empty_value:
			if gizmo is None:
				gizmo = self._gizmo
			if parse is None:
				parse = self._parse_space
			return self._StaticSpace(gizmo=gizmo, value=value, parse=parse)



class SpacedToolDecorator(ToolDecoratorBase):
	_no_space_value = object()
	def __init__(self, *args, space: Any = _no_space_value, parse: bool = None, **kwargs):
		super().__init__(*args, **kwargs)
		self._space = space
		self._parse_space = parse


	def _actualize_tool(self, fn: Callable, space=_no_space_value, parse_space=None, **kwargs):
		if space is self._no_space_value:
			space = self._space
		if parse_space is None:
			parse_space = self._parse_space
		if space is self._no_space_value:
			return super()._actualize_tool(fn, **kwargs)
		return super()._actualize_tool(fn, space=space, parse_space=parse_space, **kwargs)




class Indicator(SpacedToolDecorator):
	def __init__(self, *gizmos, space=SpacedToolDecorator._no_space_value, parse=None, **kwargs):
		if space is self._no_space_value:
			space = Scalar()
		if parse is None:
			parse = False
		super().__init__(*gizmos, space=space, parse=parse, **kwargs)




