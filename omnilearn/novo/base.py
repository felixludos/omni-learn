

from omnifig import script, component, creator, modifier
from omnidata.framework import BuilderCreator as _BuilderCreator, register_builder as _register_builder

creator('build')(_BuilderCreator)


class builder(_register_builder):
	def __init__(self, *args, description=None, creator='build', **kwargs):
		super().__init__(*args, **kwargs)
		self.description = description
		self.creator = creator

	def __call__(self, obj):
		name = self.params.get('name', None)
		if name is not None:
			component(name, creator=self.creator, description=self.description)(obj)
		return super().__call__(obj)

















