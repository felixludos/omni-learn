from typing import Sequence, Union, Any, Optional, Tuple, Dict, List, Iterable, Iterator, Callable, Type, Set
from torch import nn

from omnibelt import agnostic, Class_Registry
from omnifig import script, component, creator, modifier

from omniplex import hparam, inherit_hparams, submodule, submachine, material, space, indicator, machine, \
	Structured, Builder, spaces, Spec

from omniplex import Named, BuildCreator as _BuilderCreator, register_builder as _register_builder, \
	HierarchyBuilder as _HierarchyBuilder, RegisteredProduct as _RegisteredProduct
from omniplex import get_builder

from omniplex.tools import Signature


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



# dataset_registry = Class_Registry()
# get_dataset = dataset_registry.get_class
# class register_dataset(component):
# 	@classmethod
# 	def register(cls, name, item, **kwargs) -> None:
# 		dataset_registry.new(name, item, **kwargs)
# 		return super().register(name, item, **kwargs)



class BranchBuilder(_HierarchyBuilder, create_registry=False):
	# automatically register builders as components (with the creator `build`)
	def __init_subclass__(cls, branch=None, create_registry=None, description=None,
	                      skip_component_registration=False, **kwargs):
		super().__init_subclass__(branch=branch, **kwargs)
		if not skip_component_registration and (create_registry or branch is not None):
			builder(cls.hierarchy_registry_address(), description=description)(cls)


	@classmethod
	def register_product(cls, name, product, is_default=False, *, creator='build', description=None, **kwargs):
		addr = cls.hierarchy_registry_address()
		if addr is not None:
			if description is None:
				description = getattr(product, 'description', None)
			component(f'{addr}{cls._branch_address_delimiter}{name}', creator=creator, description=description)(product)
		return super().register_product(name, product, **kwargs)



class WorldBuilder(BranchBuilder, branch='world'):
	pass



class DataBuilder(WorldBuilder, branch='data'):
	pass



class ModelBuilder(WorldBuilder, branch='model'):
	pass



class Product(_RegisteredProduct):
	pass



class DataProduct(Product, registry='data'):
	pass



class ModelProduct(Product, registry='model'):
	pass



class Blueprint(Spec):
	def _fix_space(self, space):
		builder = get_builder('space')()
		return builder.validate(space)


	def change_space_of(self, gizmo: str, space: spaces.Dim):
		space = self._fix_space(space)
		return super().change_space_of(gizmo, space)



class SpaceBuilder(WorldBuilder, branch='space', products={
	'unbound': spaces.Unbound,
	'categorical': spaces.Categorical
}):
	def validate(self, product):
		if isinstance(product, (int, tuple)):
			return self.build('unbound', shape=product)
		if isinstance(product, Sequence):
			return self.build('categorical', n=product)
		return super().validate(product)




