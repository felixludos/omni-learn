from .imports import *
from .spaced import SpacedToolDecorator, SpacedToolCraft, ShortcutToolCraft, SpaceGem, StaticSpace, Indicator

import typing
from functools import cached_property
from omnibelt import annotation_is_compatible


class hparam(omniply_gem):
	def __init__(self, default: Any = omniply_gem._no_value, *, strict: bool = False, **kwargs):
		super().__init__(default=default, **kwargs)
		self._strict = strict

	def _validate(self, value: Any):
		if self.annotation is not None and not annotation_is_compatible(value, self.annotation):
			if self._strict:
				raise TypeError(f"{self._owner.__name__}.{self._name} expected {self.annotation} but got {value!r}")
			else:
				print(f'WARNING: {self._owner.__name__}.{self._name} expected {self.annotation} but got {value!r}')

	def rebuild(self, instance: 'AbstractGeologist', value: Any):
		built = super().rebuild(instance, value)
		self._validate(built)
		return built

	# def revise(self, instance, value):
	# 	self._validate(value)
	# 	return super().revise(instance, value)

	@cached_property
	def annotation(self):
		return typing.get_type_hints(self._owner).get(self._name, None) if self._fn is None \
			else self._fn.__annotations__.get('return', None)



class submodule(omniply_geode):
	pass



class gear(omniply_gear):
	pass



class space(SpaceGem):
	pass



class tool(SpacedToolDecorator, omniply_tool):
	class _ToolCraft(SpacedToolCraft, ShortcutToolCraft, omniply_tool._ToolCraft):
		_Space = space
		_Gear = gear
	class from_context(SpacedToolDecorator, omniply_tool.from_context):
		class _ToolCraft(SpacedToolCraft, ShortcutToolCraft, omniply_tool.from_context._ToolCraft):
			_Space = space
			_Gear = gear



class indicator(Indicator, tool):
	class from_context(Indicator, tool.from_context):
		pass



