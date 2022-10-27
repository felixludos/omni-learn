

from omnifig import script, component, creator, modifier
from omnidata import Named, BuilderCreator as _BuilderCreator, register_builder as _register_builder, \
	ClassBuilder as _ClassBuilder


creator('build')(_BuilderCreator)


class builder(_register_builder): # automatically register builders as components (with the creator `build`)
	def __init__(self, *args, description=None, creator='build', **kwargs):
		super().__init__(*args, **kwargs)
		self.description = description
		self.creator = creator

	def __call__(self, obj):
		name = self.params.get('name', None)
		if name is not None:
			component(name, creator=self.creator, description=self.description)(obj)
		return super().__call__(obj)



class ClassBuilder(_ClassBuilder): # auto register classes in the class registry as components (and builders)
	def __init_subclass__(cls, ident=None, skip_component_registration=False, **kwargs):
		super().__init_subclass__(ident=ident, **kwargs)
		if ident is not None and not skip_component_registration:
			builder(ident)(cls)



class DatasetBuilder(ClassBuilder, Named): # auto register datasets as components (and builders)
	def __init_subclass__(cls, ident=None, **kwargs):
		if ident is None and cls.name is not None:
			ident = cls.name
		if ident is not None:
			cls.name = ident
			ident = f'dataset/{ident}'
		super().__init_subclass__(ident=ident, **kwargs)
	
	










