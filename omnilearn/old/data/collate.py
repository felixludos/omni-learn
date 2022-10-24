
import torch
import re
from torch._six import container_abcs, string_classes, int_classes
from omnibelt import unspecified_argument
import omnifig as fig

from omnilearn.util import TensorDict, TensorList

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def _collate_movable(batch):
	r"""Puts each data field into a tensor with outer dimension batch size"""

	elem = batch[0]
	elem_type = type(elem)
	if isinstance(elem, torch.Tensor):
		out = None
		if torch.utils.data.get_worker_info() is not None:
			# If we're in a background process, concatenate directly into a
			# shared memory tensor to avoid an extra copy
			numel = sum([x.numel() for x in batch])
			storage = elem.storage()._new_shared(numel)
			out = elem.new(storage)
		return torch.stack(batch, 0, out=out)
	elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
			and elem_type.__name__ != 'string_':
		elem = batch[0]
		if elem_type.__name__ == 'ndarray':
			# array of string classes and object
			if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
				raise TypeError(default_collate_err_msg_format.format(elem.dtype))

			return _collate_movable([torch.as_tensor(b) for b in batch])
		elif elem.shape == ():  # scalars
			return torch.as_tensor(batch)
	elif isinstance(elem, float):
		return torch.tensor(batch, dtype=torch.float64)
	elif isinstance(elem, int_classes):
		return torch.tensor(batch)
	elif isinstance(elem, string_classes):
		return batch
	elif isinstance(elem, container_abcs.Mapping):
		return TensorDict({key: _collate_movable([d[key] for d in batch]) for key in elem})
	elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
		return elem_type(*(_collate_movable(samples) for samples in zip(*batch)))
	elif isinstance(elem, container_abcs.Sequence):
		transposed = zip(*batch)
		return TensorList([_collate_movable(samples) for samples in transposed])

	raise TypeError(default_collate_err_msg_format.format(elem_type))



# class SampleFormat:
# 	def __init__(self, *names, typ=None):
# 		self.available = set(names)
# 		self.format = list(names)
# 		self.typ = typ
#
#
# 	def format_type(self, typ=unspecified_argument):
# 		if typ is not unspecified_argument:
# 			self.typ = typ
# 		return self.typ
#
#
# 	def reformat(self, *names):
# 		new, missing = [], []
# 		for name in names:
# 			(new if name in self.available else missing).append(name)
# 		if len(missing):
# 			raise MissingDataError(*missing)
# 		self.format = new
#
#
# 	def add(self, *data):
# 		self.available.update(data)
#
#
# 	def format(self, sample):
#
# 		pass
#
#
# 	def collate(self, samples):
# 		pass
#
# 	pass









