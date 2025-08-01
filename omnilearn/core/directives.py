from .imports import *
from .spaced import SpacedToolDecorator, SpacedToolCraft, ShortcutToolCraft, SpaceSkill, SpaceCraft, Indicator



class hparam(omniply_gem):
	pass



class submodule(omniply_geode):
	pass



class gear(omniply_gear):
	pass



class space(SpaceCraft, omniply_gear):
	class _GearSkill(SpaceSkill, gear._GearSkill):
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



