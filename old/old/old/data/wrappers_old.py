
import sys, os
from pathlib import Path
from wrapt import ObjectProxy
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDataset, TensorDataset
import h5py as hf

from omnibelt import unspecified_argument, InitWall

from .collectors import Supervised, Batchable

from omnilearn import util



class DatasetWrapper(ObjectProxy):

	def __init__(self, wrapped):
		super().__init__(wrapped)


	def __getattr__(self, name):
		# If we are being to lookup '__wrapped__' then the
		# '__init__()' method cannot have been called.

		if name == '__wrapped__':
			raise ValueError('wrapper has not been initialised')

		return getattr(self.__wrapped__, name)


	# 	self._self__basegetattribute__ = wrapped.__getattribute__
	# 	wrapped.__getattribute__ = self._make_deep_get(wrapped)
	#
	# def _make_deep_get(self, wrapped):
	# 	def _getattribute(item):
	# 		if item == '__basegetattribute__':
	# 			return self._self__basegetattribute__(item)
	# 		return self.__getattribute__(item)
	# 	return _getattribute




class Indexed_Dataset(DatasetWrapper):
	def __getitem__(self, idx):
		return idx, self.__wrapped__[idx]



class Subset_Dataset(DatasetWrapper):
	def __init__(self, dataset, indices=None, num=None, shuffle=True, update_data=True):
		super().__init__(dataset)
		if indices is None:
			# assert num is not None, f'{num} is needed'
			if num is not None:
				indices = torch.randperm(len(dataset))[:num] if shuffle else torch.arange(num)
		self._self_indices = indices

		try:
			device = dataset.get_device()
			if self._self_indices is not None:
				self._self_indices = self._self_indices.to(device)
		except AttributeError:
			pass

		self._self_subset_updated = False
		if update_data:
			try:
				if self._self_indices is not None:
					self.update_data(self._self_indices)
					self._self_subset_updated = True
			except AttributeError:
				print('WARNING: Subset failed to update underlying dataset automatically')
				pass

	def get_observations(self):
		if self._self_subset_updated:
			return self.__wrapped__.get_observation()
		if isinstance(self.__wrapped__, Batchable):
			return self[torch.arange(len(self))][0]
		return util.pytorch_collate([self[i][0] for i in range(len(self))])

	def get_labels(self):
		if self._self_subset_updated:
			return self.__wrapped__.get_target()
		if isinstance(self.__wrapped__, Batchable):
			return self[torch.arange(len(self))][1]
		return util.pytorch_collate([self[i][1] for i in range(len(self))])

	def __getitem__(self, idx):
		return self.__wrapped__[idx] if self._self_indices is None or self._self_subset_updated else self.__wrapped__[self._self_indices[idx]]


	def __len__(self):
		return len(self.__wrapped__) if self._self_indices is None or self._self_subset_updated else len(self._self_indices)



class Repeat_Dataset(DatasetWrapper):
	def __init__(self, dataset, factor):
		super().__init__(dataset)
		self._self_factor = factor
		self._self_num_real = len(dataset)
		self._self_total = self._self_factor * self._self_num_real
		print('Repeating dataset {} times'.format(factor))


	def __getitem__(self, idx):
		return self.__wrapped__[idx % self._self_num_real]


	def __len__(self):
		return self._self_total



class Format_Dataset(DatasetWrapper):
	def __init__(self, dataset, format_fn, format_args=None, include_original=False):
		super().__init__(dataset)

		self._self_format_fn = format_fn
		self._self_format_args = {} if format_args is None else format_args
		self._self_include_original = include_original


	def __getitem__(self, idx):

		sample = self.__wrapped__[idx]

		formatted = self._self_format_fn(sample, **self._self_format_args)

		if self._self_include_original:
			return formatted, sample

		return formatted



class Shuffle_Dataset(DatasetWrapper):
	def __init__(self, dataset):
		super().__init__(dataset)

		self._self_indices = torch.randperm(len(self.__wrapped__)).clone()

		try:
			device = self.__wrapped__.get_device()
			self._self_indices = self._self_indices.to(device).clone()
		except AttributeError:
			pass


	def __getitem__(self, idx):
		return self.__wrapped__[self._self_indices[idx]]



class SingleLabelDataset(DatasetWrapper):
	def __init__(self, dataset, idx):
		if not isinstance(dataset, Supervised):
			raise NotImplementedError
		super().__init__(dataset)
		self._self_idx = idx

		self.dout = 1
		self._subselect_info('_all_label_names', idx)
		self._subselect_info('_all_mechanism_names', idx)
		self._subselect_info('_all_mechanism_class_names', idx)
		self._subselect_info('_full_mechanism_space', idx)
		self._subselect_info('_full_label_space', idx)


	def get_label_names(self):
		return self._self__all_label_names

	def get_label_space(self):
		return self._self__all_label_space

	def get_mechanism_class_names(self, mechanism):
		if isinstance(mechanism, str):
			return self.get_mechanism_class_names(self.get_mechanism_names().index(mechanism))
		if self._self__all_mechanism_class_names is not None:
			return self._self__all_mechanism_class_names[mechanism]
	def get_mechanism_names(self):
		return self._self__all_mechanism_names
	def get_mechanism_space(self):
		if self._self__full_mechanism_space is None:
			return self.get_label_space()
		return self._self__full_mechanism_space


	def _subselect_info(self, attrname, idx):
		try:
			if getattr(self, attrname, None) is not None:
				setattr(self, f'_self_{attrname}', getattr(self, attrname)[idx])
			else:
				setattr(self, f'_self_{attrname}', None)
		except IndexError:
			pass


	def get_labels(self):
		return self.__wrapped__.get_target().narrow(-1, self._self_idx, 1)



def resolve_wrappers(ident, **kwargs):

	if not isinstance(ident, str):
		return ident

	if ident == 'single-label':
		return SingleLabelDataset
	if ident == 'shuffle':
		return Shuffle_Dataset
	if ident == 'format':
		return Format_Dataset
	if ident == 'repeat':
		return Repeat_Dataset
	if ident == 'subset':
		return Subset_Dataset
	if ident == 'indexed':
		return Indexed_Dataset

	raise Exception(f'wrapper not found: {ident}')



