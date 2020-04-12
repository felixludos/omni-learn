
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.dataloader import re, numpy_type_map, _use_shared_memory, string_classes, int_classes#,
import re
try:
	from torch.utils.data.dataloader import container_abcs
except:
	pass

from .misc import NS
import random
import collections
import itertools
import h5py as hf


class make_infinite(DataLoader):
	def __init__(self, loader):
		assert len(loader) >= 1, 'has no len'
		self.loader = loader
		self.itr = None
	
	def __len__(self):
		return len(self.loader)
	
	def end(self):
		self.itr = None
	
	def __next__(self):
		if self.itr is not None:
			try:
				return next(self.itr)
			except StopIteration:
				pass
		return next(iter(self))
	
	def __iter__(self):
		self.itr = iter(self.loader)
		return self
	


def get_h5_data(path, key, idx=None):
	with hf.File(path, 'r') as f:
		return f[key].value if idx is None else f[key][idx]


def make_np_wrapped_env(env_type):
	def create(*args, **kwargs):
		return Numpy_Env_Wrapper(env_type(*args, **kwargs))
	return create
class Numpy_Env_Wrapper(object):
	def __init__(self, env, device='cpu'):
		self._env = env
		self.device = device
		
	def to(self, d):
		self.device = d
	def cpu(self):
		self.device = 'cpu'
	def cuda(self, i=0):
		self.device = 'cuda:{}'.format(i)

	def __getattr__(self, item):
		if item not in self.__dict__:
			return self._env.__getattribute__(item)
		return super(Numpy_Env_Wrapper, self).__getattribute__(item)

	def reset(self, state=None):
		if state is None:
			return torch.from_numpy(self._env.reset()).float().to(self.device)
		return torch.from_numpy(self._env.reset(state.detach().cpu().numpy())).float().to(self.device)

	def step(self, action):
		ns, r, d, info = self._env.step(action.detach().cpu().numpy())
		ns = torch.from_numpy(ns).float().to(self.device)
		r = torch.tensor(r).float().to(self.device)
		return ns, r, d, info

	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs) # no wrapping here



def batches(iterable, n): # batches an iterable (batches are not lazy)
	iterable = iter(iterable)
	while True:
		tup = tuple(itertools.islice(iterable, 0, n))
		if tup:
			yield tup
		else:
			break

def to_one_hot(idx, max_idx=None):
	if max_idx is None:
		max_idx = idx.max()
	dims = (max_idx,)
	if idx.ndimension() >= 1:
		if idx.size(-1) != 1:
			idx = idx.unsqueeze(-1)
		dims = idx.size()[:-1] + dims
	return torch.zeros(*dims).to(idx.device).scatter_(-1, idx.long(), 1)

# linear - discretize continuous values into N classes
# TODO: implement log space
def discretize(input, N, range=None):

	if range is None:
		range = input.min(), input.max()

	input = input.clamp(*range)
	input -= range[0]

	return input.mul((N-1)/(range[1]-range[0])).round().long()



#########################
# Util to convert images to strings for compressed saving

def rgb_to_str(img, compression=5):
	assert img.shape[2] == 3
	assert img.dtype == np.uint8

	return cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compression])[1].tostring()


def float32_img_to_str(img, compression=5):
	H, W = img.shape
	assert img.dtype == np.float32

	img.dtype = np.uint8
	img.shape = (H, W, 4)

	s = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compression])[1].tostring()

	img.shape = (H, 4 * W)
	img.dtype = np.float32

	return s


def byte_img_to_str(img, compression=5):
	assert img.dtype == np.uint8

	return cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compression])[1].tostring()


def str_to_rgb(s):
	return cv2.imdecode(np.frombuffer(s, np.uint8), -1)


def str_to_float32_img(s):
	img = cv2.imdecode(np.frombuffer(s, np.uint8), -1)

	H, W = img.shape[:2]

	img = img.reshape(H, W * 4)
	img.dtype = np.float32

	return img


def str_to_byte_img(s):
	return cv2.imdecode(np.frombuffer(s, np.uint8), -1)


def jpeg_to_str(path):
	with open(path, 'rb') as f:
		return f.read()

def str_to_jpeg(s, ret_PIL=False):
	img = Image.open(BytesIO(s))
	if ret_PIL:
		return img
	return np.array(img)


# Wrapper to create a DataLoader with NS_collate
def loader(dataset, def_type=None, multi_agent=False, **kwargs): # switches default_collate to NS_collate
	if 'collate_fn' not in kwargs:
		kwargs['collate_fn'] = make_collate(def_type, multi_agent)
	if 'batch_size' in kwargs and kwargs['batch_size'] is None:
		kwargs['batch_size'] = len(dataset)
	return DataLoader(dataset, **kwargs)



# multi traj gen: [N] x {NS} x [A]  --> [A] x {NS} x [N]
#

def make_collate(stack=True):
	merge_fn = torch.stack if stack else torch.cat

	def collate(batch):
		r"""Puts each data field into a tensor with outer dimension batch size"""

		error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
		elem_type = type(batch[0])
		if isinstance(batch[0], torch.Tensor):
			out = None
			if True:
				# If we're in a background process, concatenate directly into a
				# shared memory tensor to avoid an extra copy
				numel = sum([x.numel() for x in batch])
				storage = batch[0].storage()._new_shared(numel)
				out = batch[0].new(storage)
			try:
				return merge_fn(batch, 0, out=out)
			except RuntimeError:
				return batch
		elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
				and elem_type.__name__ != 'string_':
			elem = batch[0]
			if elem_type.__name__ == 'ndarray':
				# array of string classes and object
				if re.search('[SaUO]', elem.dtype.str) is not None:
					raise TypeError(error_msg.format(elem.dtype))

				return torch.stack([torch.from_numpy(b) for b in batch], 0)
			if elem.shape == ():  # scalars
				# py_type = float if elem.dtype.name.startswith('float') else int
				# return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
				return torch.tensor(batch)
		elif isinstance(batch[0], int):
			return torch.LongTensor(batch)
		elif isinstance(batch[0], float):
			return torch.DoubleTensor(batch)
		elif isinstance(batch[0], str):
			return batch
		elif isinstance(batch[0], container_abcs.Mapping):
			return {key: collate([d[key] for d in batch]) for key in batch[0]}
		elif isinstance(batch[0], container_abcs.Sequence):
			transposed = zip(*batch)
			return [collate(samples) for samples in transposed]

		raise TypeError((error_msg.format(type(batch[0]))))

	return collate



def NS_make_collate(def_type=None, multi_agent=False):

	collate_fn = None

	if def_type is None:
		collate_fn = NS_list_collate
		return collate_fn

	def NS_list_type_collate(batch):
		r"""Puts each data field into a tensor with outer dimension batch size"""

		full = NS()

		for key in batch[0].keys():
			full[key] = []
			for sample in batch:
				data = NS_collate(sample[key]) if key == 'env_infos' else NS_collate([sample[key]])[0] # TODO: clean up
				if isinstance(data, torch.Tensor):
					data = data.type(def_type)
				full[key].append(data)

		return full  # returns NS containing collated paths

	collate_fn = NS_list_type_collate

	if multi_agent:
		if def_type is None:
			return NS_multi_list_collate

		def NS_multi_list_type_collate(batch):  # no default tensor type
			r"""Puts each data field into a tensor with outer dimension batch size"""

			num_agents = len(next(iter(batch[0].values())))

			full = [NS() for _ in range(num_agents)]  # a separate batch of paths for each agent

			for i, f in enumerate(full):
				for key in batch[0].keys():
					for sample in batch:
						if isinstance(sample[key], list):
							if isinstance(sample[key][0], dict):
								data = NS_collate(sample[key])
							else:
								data = NS_collate([sample[key][i]])[0]
								if isinstance(data, torch.Tensor):
									data = data.type(def_type)
						else:
							data = NS_collate([sample[key]])[0]
							if isinstance(data, torch.Tensor):
								data = data.type(def_type)
						if key not in f:
							f[key] = []
						f[key].append(data)

			#print(full[0].rewards)

			return full  # returns NS containing collated paths

		return NS_multi_list_type_collate

	return collate_fn


def NS_multi_list_collate(batch):  # no default tensor type
	r"""Puts each data field into a tensor with outer dimension batch size"""

	num_agents = len(next(batch[0].values()))

	full = [NS() for _ in range(num_agents)] # a separate batch of paths for each agent

	for i, f in enumerate(full):
		for key in batch[0].keys():
			f[key] = [NS_collate([sample[key][i]])[0] for sample in batch]

	return full  # returns NS containing collated paths

def NS_list_collate(batch): # no default tensor type
	r"""Puts each data field into a tensor with outer dimension batch size"""
	
	full = NS()
	
	if isinstance(batch, dict):
		full = NS(**{k:NS_list_collate(batch[k]) for k in batch.keys()})
		return full
	
	for key in batch[0].keys():
		full[key] = [NS_collate([sample[key]])[0] for sample in batch]
		
	return full # returns NS containing collated paths

def NS_collate(batch):
	r"""Puts each data field into a tensor with outer dimension batch size"""

	error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
	
	if isinstance(batch, dict):
		return {k:NS_collate(batch[k]) for k in batch.keys()}
	
	elem_type = type(batch[0])
	if isinstance(batch[0], torch.Tensor):
		out = None
		if _use_shared_memory:
			# If we're in a background process, concatenate directly into a
			# shared memory tensor to avoid an extra copy
			numel = sum([x.numel() for x in batch])
			storage = batch[0].storage()._new_shared(numel)
			out = batch[0].new(storage)
		return torch.stack(batch, 0, out=out)
	elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
			and elem_type.__name__ != 'string_':
		elem = batch[0]
		if elem_type.__name__ == 'ndarray':
			# array of string classes and object
			if re.search('[SaUO]', elem.dtype.str) is not None:
				raise TypeError(error_msg.format(elem.dtype))

			return torch.stack([torch.from_numpy(b) for b in batch], 0)
		if elem.shape == ():  # scalars
			py_type = float if elem.dtype.name.startswith('float') else int
			return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
	elif isinstance(batch[0], int_classes):
		return torch.LongTensor(batch)
	elif isinstance(batch[0], float):
		return torch.DoubleTensor(batch)
	elif isinstance(batch[0], string_classes):
		return batch
	elif isinstance(batch[0], NS):
		return NS(**{key: NS_collate([d[key] for d in batch]) for key in batch[0]})
	elif isinstance(batch[0], collections.Mapping):
		return {key: NS_collate([d[key] for d in batch]) for key in batch[0]}
	elif isinstance(batch[0], torch.distributions.Distribution):
		return [distrib for distrib in batch]
	elif isinstance(batch[0], collections.Sequence):
		transposed = zip(*batch)
		return [NS_collate(samples) for samples in transposed]

	raise TypeError((error_msg.format(type(batch[0]))))