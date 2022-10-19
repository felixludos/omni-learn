
import sys, os, types
from pathlib import Path
from wrapt import ObjectProxy
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset, TensorDataset
import h5py as hf

from omnibelt import unspecified_argument, InitWall, Class_Registry, \
	mix_into, wrap_class, conditional_method, duplicate_instance, duplicate_func, join_classes, replace_class
import omnifig as fig

from .collectors import Observation, Supervised, Batchable, Lockable, DatasetBase, Dataset, Disentanglement, \
	MechanisticBase, ListDataset

from .. import util



class Wrapper_Registry(Class_Registry, components=['activate_fn', 'rewrappable']):
	class DecoratorBase(Class_Registry.DecoratorBase):
		def _register(self, val, name=None, activate_fn=None, rewrappable=False, **params):
			if name is None:
				name = val.__name__

			if activate_fn is None:
				activate_fn = getattr(val, '__activate__', None)

			if rewrappable:
				def _chain_wrapper(base):
					return wrap_class(val, base, chain=False)
				# fig.Modifier(name)(_chain_wrapper)
			else:
				# fig.AutoModifier(name)(val)
				pass

			entry = super()._register(val, name=name, activate_fn=activate_fn, rewrappable=rewrappable, **params)

			for attr in val.__dict__.values():
				if isinstance(attr, self.condition):
					attr.wrapper(entry)
			return entry


		class condition(conditional_method):
			def __init__(self, *bases):
				super().__init__()
				self.bases = bases
				self.entry = None

			# def _build_method(self, fn, instance, owner):
			# 	return types.MethodType(duplicate_func(fn, cls=owner.__unwrapped__), instance)

			def wrapper(self, entry):
				self.entry = entry

			def condition(self, instance):
				return any(isinstance(instance, cond) for cond in self.bases)

			# def _build_method(self, instance, owner):
			# 	return types.MethodType(duplicate_func(self.fn, cls=owner.__bases__[-1]), instance)



wrapper_registry = Wrapper_Registry()
DatasetWrapper = wrapper_registry.get_decorator('DatasetWrapper')


class AlreadyWrappedError(Exception):
	def __init__(self, wrapper, obj):
		super().__init__(f'The object {obj} is already wrapped with wrapper "{wrapper.name}"')
		self.wrapper = wrapper
		self.obj = obj


def wrap_dataset(wrapper, dataset, *args, **kwargs):
	if not isinstance(wrapper, wrapper_registry.entry_cls):
		wrapper = wrapper_registry.find(wrapper)

	base = dataset.__class__
	cls = wrapper.cls

	if wrapper.rewrappable:
		new = wrap_class(cls, base, chain=False)
	elif isinstance(dataset, cls):
		raise AlreadyWrappedError(wrapper, dataset)
	else:
		new = join_classes(cls, base)
	new.__unwrapped__ = base
	obj = replace_class(duplicate_instance(dataset), new)

	if wrapper.activate_fn is not None:
		wrapper.activate_fn(obj, *args, **kwargs)
	return obj



#@DatasetWrapper('indexed')
class Indexed(DatasetBase):
	def __getitem__(self, idx):
		return idx, super().__getitem__(idx)



#@DatasetWrapper('shuffled')
class Shuffled(DatasetBase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		Shuffled.__activate__(self)


	def __activate__(self):
		indices = torch.randperm(len(self))
		if isinstance(self, Lockable):
			self._update_data(indices)
			self._is_shuffled = True
		else:
			self._is_shuffled = False
			self._shuffle_indices = indices


	def get(self, name=None, idx=None, **kwargs):
		if not self._is_shuffled and idx is not None:
			idx = self._shuffle_indices[idx]
		return super().get(name=name, idx=idx, **kwargs)

	# def __getitem__(self, idx):
	# 	return super().__getitem__(idx if self._is_shuffled else self._self_indices[idx])



#@DatasetWrapper('subset', rewrappable=True)
class Subset(Dataset):
	def __init__(self, A, indices=unspecified_argument, num=unspecified_argument, shuffle=unspecified_argument,
	             update_data=unspecified_argument, **kwargs):

		if indices is unspecified_argument:
			indices = A.pull('indices', None)

		if num is unspecified_argument:
			num = A.pull('num', None)

		if shuffle is unspecified_argument:
			shuffle = A.pull('shuffle-subset', True)

		if update_data is unspecified_argument:
			update_data = A.pull('update-data', True)

		super().__init__(A, **kwargs)
		Subset.__activate__(self, indices=indices, num=num, shuffle=shuffle, update_data=update_data)


	def __activate__(self, indices=None, num=None, shuffle=True, update_data=True):
		if indices is None and num is not None:
			indices = torch.randperm(len(self))[:num] if shuffle else torch.arange(num)

		if isinstance(self, Lockable):
			# self._subset_selected = True
			self._update_data(indices)
			indices = None
		else:
			# self._subset_selected = False
			prev = getattr(self, '_subset_indices', None)
			if prev is not None:
				indices = prev[indices]

		self._subset_indices = indices


	# #@DatasetWrapper.condition(Observation)
	# def get_observations(self, idx=None):
	# 	if self._subset_indices is None:
	# 		return super().get_observations(idx=idx)
	# 	if isinstance(self, Batchable):
	# 		return self[torch.arange(len(self))][0]
	# 	return util.pytorch_collate([self[i][0] for i in range(len(self))])
	#
	#
	# #@DatasetWrapper.condition(Supervised)
	# def get_targets(self, idx=None):
	# 	if self._subset_indices is None:
	# 		return super().get_targets(idx=idx)
	# 	if isinstance(self.__wrapped__, Batchable):
	# 		return self[torch.arange(len(self))][1]
	# 	return util.pytorch_collate([self[i][1] for i in range(len(self))])


	# def __getitem__(self, idx):
	# 	return super().__getitem__(idx if self._subset_indices is None else self._subset_indices[idx])


	# def __len__(self):
	# 	return super().__len__() if self._subset_indices is None else len(self._subset_indices)



#@DatasetWrapper('single-label')
class SingleLabel(Dataset):
	def __init__(self, A, idx=None, **kwargs):
		if idx is None:
			idx = A.pull('label-idx')

		super().__init__(A, **kwargs)
		SingleLabel.__activate__(self, idx)


	def __activate__(self, idx):

		targets = self.get('targets').narrow(-1, idx, 1)
		self._available_data_keys = self._available_data_keys.copy()
		self.register_data('targets', data=targets)

		# housekeeping

		if isinstance(self, Supervised):
			if self._target_space is not None:
				self._target_space = self._target_space[idx]
			if self._target_names is not None:
				self._target_names = self._target_names[idx]
		if isinstance(self, MechanisticBase):
			if self._full_mechanism_space is not None:
				self._full_mechanism_space = self.get_mechanism_class_space(idx)
		if isinstance(self, Disentanglement):
			names = self.get_label_class_names(idx)
			space = self.get_label_class_space(idx)
			if self._all_label_names is not None:
				self._all_label_names = names
			if self._all_label_class_names is not None:
				self._all_label_class_names = names
			if self._full_label_space is not None:
				self._full_label_space = space

		self._label_idx = idx
		self.dout = 1




	# #@DatasetWrapper.condition(Supervised)
	# def get_targets(self, idx=None):
	# 	return super().get_targets(idx=idx).narrow(-1, self._label_idx, 1)


	# def __getitem__(self, item):
	# 	obs, *other = super().__getitem__(item)
	# 	if len(other):
	# 		lbl, *other = other
	# 		return obs, lbl[self._label_idx], *other
	# 	return obs,



# #@DatasetWrapper('repeated')
class Repeated(Dataset):
	def __init__(self, A, factor=unspecified_argument, **kwargs):

		raise NotImplementedError

		if factor is unspecified_argument:
			factor = A.pull('repeat-factor', None)

		super().__init__(A, **kwargs)
		Repeated.__activate__(self, factor=factor)


	def __activate__(self, factor=None):
		if hasattr(self, '_repeat_factor'):
			prev = self._repeat_factor
			self._repeat_factor = factor

		self._repeat_num_real = None


	# def __init__(self, dataset, factor):
	# 	super().__init__(dataset)
	# 	self._self_factor = factor
	# 	self._self_num_real = len(dataset)
	# 	self._self_total = self._self_factor * self._self_num_real
	# 	print('Repeating dataset {} times'.format(factor))


	def __getitem__(self, idx):
		return self.__wrapped__[idx % self._self_num_real]


	def __len__(self):
		return self._self_total



# #@DatasetWrapper('format')
class Formatted(Dataset):
	def __init__(self, A, format_fn=unspecified_argument, format_args=None, include_original=None, **kwargs):

		raise NotImplementedError

		if format_fn is unspecified_argument:
			format_fn = A.pull('format-fn', None)

		if format_args is None:
			format_args = A.pull('format-args', {})

		if include_original is None:
			include_original = A.pull('include-original', False)

		super().__init__(A, **kwargs)


	def __activate__(self, format_fn=None, format_args=None, include_original=False):

		self._format_fn = format_fn
		self._format_args = {} if format_args is None else format_args
		self._include_original = include_original


	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		formatted = sample if self._format_fn is None else self._format_fn(sample, **self._format_args)

		if self._include_original:
			return formatted, sample
		return formatted



# class DatasetWrapper(ObjectProxy):
#
# 	def __init__(self, wrapped):
# 		super().__init__(wrapped)
#
#
# 	def __getattr__(self, name):
# 		# If we are being to lookup '__wrapped__' then the
# 		# '__init__()' method cannot have been called.
#
# 		if name == '__wrapped__':
# 			raise ValueError('wrapper has not been initialised')
#
# 		return getattr(self.__wrapped__, name)
#
#
# 	# 	self._self__basegetattribute__ = wrapped.__getattribute__
# 	# 	wrapped.__getattribute__ = self._make_deep_get(wrapped)
# 	#
# 	# def _make_deep_get(self, wrapped):
# 	# 	def _getattribute(item):
# 	# 		if item == '__basegetattribute__':
# 	# 			return self._self__basegetattribute__(item)
# 	# 		return self.__getattribute__(item)
# 	# 	return _getattribute




# class Indexed_Dataset(DatasetWrapper):
# 	def __getitem__(self, idx):
# 		return idx, self.__wrapped__[idx]
#
#
#
# class Subset_Dataset(DatasetWrapper):
# 	def __init__(self, dataset, indices=None, num=None, shuffle=True, update_data=True):
# 		super().__init__(dataset)
# 		if indices is None:
# 			# assert num is not None, f'{num} is needed'
# 			if num is not None:
# 				indices = torch.randperm(len(dataset))[:num] if shuffle else torch.arange(num)
# 		self._self_indices = indices
#
# 		try:
# 			device = dataset.get_device()
# 			if self._self_indices is not None:
# 				self._self_indices = self._self_indices.to(device)
# 		except AttributeError:
# 			pass
#
# 		self._self_subset_updated = False
# 		if update_data:
# 			try:
# 				if self._self_indices is not None:
# 					self.update_data(self._self_indices)
# 					self._self_subset_updated = True
# 			except AttributeError:
# 				print('WARNING: Subset failed to update underlying dataset automatically')
# 				pass
#
# 	def get_observations(self):
# 		if self._self_subset_updated:
# 			return self.__wrapped__.get_observations()
# 		if isinstance(self.__wrapped__, Batchable):
# 			return self[torch.arange(len(self))][0]
# 		return util.pytorch_collate([self[i][0] for i in range(len(self))])
#
# 	def get_labels(self):
# 		if self._self_subset_updated:
# 			return self.__wrapped__.get_labels()
# 		if isinstance(self.__wrapped__, Batchable):
# 			return self[torch.arange(len(self))][1]
# 		return util.pytorch_collate([self[i][1] for i in range(len(self))])
#
# 	def __getitem__(self, idx):
# 		return self.__wrapped__[idx] if self._self_indices is None or self._self_subset_updated else self.__wrapped__[self._self_indices[idx]]
#
#
# 	def __len__(self):
# 		return len(self.__wrapped__) if self._self_indices is None or self._self_subset_updated else len(self._self_indices)
#
#
#
# class Repeat_Dataset(DatasetWrapper):
# 	def __init__(self, dataset, factor):
# 		super().__init__(dataset)
# 		self._self_factor = factor
# 		self._self_num_real = len(dataset)
# 		self._self_total = self._self_factor * self._self_num_real
# 		print('Repeating dataset {} times'.format(factor))
#
#
# 	def __getitem__(self, idx):
# 		return self.__wrapped__[idx % self._self_num_real]
#
#
# 	def __len__(self):
# 		return self._self_total
#
#
#
# class Format_Dataset(DatasetWrapper):
# 	def __init__(self, dataset, format_fn, format_args=None, include_original=False):
# 		super().__init__(dataset)
#
# 		self._self_format_fn = format_fn
# 		self._self_format_args = {} if format_args is None else format_args
# 		self._self_include_original = include_original
#
#
# 	def __getitem__(self, idx):
#
# 		sample = self.__wrapped__[idx]
#
# 		formatted = self._self_format_fn(sample, **self._self_format_args)
#
# 		if self._self_include_original:
# 			return formatted, sample
#
# 		return formatted
#
#
#
# class Shuffle_Dataset(DatasetWrapper):
# 	def __init__(self, dataset):
# 		super().__init__(dataset)
#
# 		self._self_indices = torch.randperm(len(self.__wrapped__)).clone()
#
# 		try:
# 			device = self.__wrapped__.get_device()
# 			self._self_indices = self._self_indices.to(device).clone()
# 		except AttributeError:
# 			pass
#
#
# 	def __getitem__(self, idx):
# 		return self.__wrapped__[self._self_indices[idx]]



# def resolve_wrappers(ident, **kwargs):
#
# 	if not isinstance(ident, str):
# 		return ident
#
# 	if ident == 'single-label':
# 		return SingleLabelDataset
# 	if ident == 'shuffle':
# 		return Shuffle_Dataset
# 	if ident == 'format':
# 		return Format_Dataset
# 	if ident == 'repeat':
# 		return Repeat_Dataset
# 	if ident == 'subset':
# 		return Subset_Dataset
# 	if ident == 'indexed':
# 		return Indexed_Dataset
#
# 	raise Exception(f'wrapper not found: {ident}')



