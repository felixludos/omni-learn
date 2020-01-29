

import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .. import util
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from ..data.loaders import BatchedDataLoader
from ..data.collectors import *
from ..data.collate import _collate_movable
from .registry import create_component, Component
from .config import get_config

# _dataset_registry = {}
# _testable_registry = set()
#
# def register_dataset(name, cls, *args, **kwargs):
# 	_dataset_registry[name] = cls, args, kwargs
# 	if issubclass(cls, Testable_Dataset):
# 		_testable_registry.add(name)


FD_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(FD_PATH),'local_data')
# print(FD_PATH)


def get_loaders(*datasets, batch_size=64, num_workers=0, shuffle=True, pin_memory=True,
		   drop_last=False, worker_init_fn=None, silent=False, allow_batched=True):

	if shuffle == 'all':
		shuffles = [True]*3
	elif shuffle:
		shuffles = [True, False, False]
	else:
		shuffles = [False]*3

	ds = datasets[0]

	loader_cls = DataLoader
	kwargs = {
		'batch_size': batch_size,
		'drop_last': drop_last,
	}

	if allow_batched:
		try:
			assert ds.allow_batched()
		except (AttributeError, AssertionError):
			pass
		else:
			print('Using batched data loader')
			loader_cls = BatchedDataLoader
	else:

		try:
			assert ds.get_device() == 'cpu'
		except AttributeError:
			pass
		except AssertionError:
			pin_memory = False

		kwargs.update({
			'pin_memory': pin_memory,
			'worker_init_fn': worker_init_fn,
			'num_workers': num_workers,
		})


	loaders = [loader_cls(ds, shuffle=s, **kwargs) for ds, s in zip(datasets, shuffles)]

	# if not silent: # TODO: deprecated!
	# 	trainloader = loaders[0]
	# 	testloader = None if len(loaders) < 2 else loaders[-1]
	# 	valloader = None if len(loaders) < 3 else loaders[1]
	#
	# 	print('traindata len={}, trainloader len={}'.format(len(datasets[0]), len(trainloader)))
	# 	if valloader is not None:
	# 		print('valdata len={}, valloader len={}'.format(len(datasets[1]), len(valloader)))
	# 	if testloader is not None:
	# 		print('testdata len={}, testloader len={}'.format(len(datasets[-1]), len(testloader)))
	# 	print('Batch size: {} samples'.format(batch_size))

	if len(loaders) == 1:
		return loaders[0]
	return loaders


def simple_split_dataset(dataset, split, shuffle=True):
	'''

	:param dataset:
	:param split: split percent as ratio [0,1]
	:param shuffle:
	:return:
	'''

	assert 0 < split < 1

	if shuffle:
		dataset = Shuffle_Dataset(dataset)

	ncut = int(len(dataset) * split)

	part1 = Subset_Dataset(dataset, torch.arange(0,ncut))
	part2 = Subset_Dataset(dataset, torch.arange(ncut, len(dataset)))

	return part1, part2

def split_dataset(dataset, split1, split2=None, shuffle=True):
	p1, p2 = simple_split_dataset(dataset, split1, shuffle=shuffle)
	if split2 is None:
		return p1, p2
	split2 = split2 / (1 - split1)
	p2, p3 = simple_split_dataset(p2, split2, shuffle=False)
	return p1, p2, p3

_dataset_registry = {}
def register_dataset(name, dataset):
	assert name[:8] != 'dataset/', 'invalid name: {}'.format(name)

	cmpn_name = 'dataset/{}'.format(name)

	Component(cmpn_name)(dataset)

	_dataset_registry[name] = cmpn_name

def view_dataset_registry():
	return _dataset_registry.copy()

#
# @Component('dataset')
# class _Dataset(object):
# 	def __new__(cls, A):
#
# 		name = A.pull('name')


def Dataset(name): # WARNING: datasets should not pull "name"
	def _dataset(dataset):
		nonlocal name

		register_dataset(name, dataset)
		return dataset

	return _dataset


def get_dataset(name=None, mode=None, info=None, **kwargs):

	assert name is not None or info is not None, 'must provide either name (+ **kwargs) or config ("info")'

	created_config = False
	if info is None:
		created_config = True
		info = get_config()

		print('Loading dataset: {}'.format(name))

		info.name = name
		info.update(kwargs)


	if '_type' not in info:
		assert 'name' in info, 'no dataset specified'
		info._type = _dataset_registry[info.name]

	if 'dataroot' not in info and 'FOUNDATION_DATA_DIR' in os.environ:
		info.dataroot = os.environ['FOUNDATION_DATA_DIR']

	if mode is not None:  # TODO: doesn't allow for other modes
		info.train = mode == 'train'

	if created_config:
		info.begin()

	dataset = create_component(info)

	if created_config:
		info.abort()

	return dataset

class UntestableDatasetError(Exception):
	def __init__(self, name):
		super().__init__('Unable to only load testset for {}, since it doesnt subclass Testable_Dataset'.format(name))


def default_load_data(info, mode='train'):
	'''
	req: A.dataset, A.device
	optional: A.dataset[mode], A.dataset.device
	adds: A.data, A.model.din, A.model.dout
	:param info:
	:return:
	'''

	# assert '_type' in info

	# info = A.dataset
	# if mode in info:
	# 	info = info[mode]

	# print('Dataset: {}'.format(info._type))

	# dataset = create_component(info)



	# if '_type' in info:
	# 	if mode == 'test':  # TODO: maybe too heavy handed
	# 		info.train = False
	#
	# 	dataset = create_component(info)
	# elif 'name' in info:
	# 	name = info.pull('name')
	# 	dataset = get_dataset(name, mode=mode, info=info)
	# else:
	# 	raise Exception('Unable to create dataset without either _type or name')

	dataset = get_dataset(info=info, mode=mode)

	assert 'device' in info, 'No device selected'
	device = info.device
	try:
		dataset.to(device)
		print('Dataset moved to {}'.format(device))
	except AttributeError:
		pass
	except RuntimeError:
		print('Not enough memory to move dataset to {}'.format(device))

	# if mode == 'test':
	# 	return dataset

	return dataset

#
# def default_load_data(A, mode='train'):
# 	'''
# 	req: A.dataset, A.device
# 	optional: A.dataset[mode], A.dataset.device
# 	adds: A.data, A.model.din, A.model.dout
# 	:param A:
# 	:return:
# 	'''
#
# 	info = A.dataset
# 	if mode in info:
# 		info = info[mode] # TODO: merge A.dataset[mode] with defaults in A.dataset
#
# 	name = info.pull('name')
# 	args = info.args if 'args' in info else ()
# 	kwargs = dict(info.kwargs) if 'kwargs' in info else {}
#
# 	if 'dataroot' in info:
# 		kwargs['dataroot'] = info.dataroot
#
# 	if mode == 'test':
# 		if name not in _testable_registry:
# 			raise UntestableDatasetError(name)
# 		kwargs['train'] = False
#
# 	dataset = get_dataset(name, *args, **kwargs)
#
# 	assert 'device' in info, 'No device selected'
# 	device = info.device
# 	try:
# 		dataset.to(device)
# 		print('Dataset {} moved to {}'.format(name, device))
# 	except AttributeError:
# 		pass
# 	except RuntimeError:
# 		print('Not enough memory to move dataset to {}'.format(device))
#
# 	if mode == 'test':
# 		return dataset
#
# 	try:
# 		din, dout = dataset.get_info()
# 		A.din, A.dout = din, dout
# 		print('Dataset din={}, dout={}'.format(din, dout))
# 	except AttributeError:
# 		pass
#
# 	trainsets = dataset,
# 	testset = None
# 	if 'test_split' in info:
# 		assert 0 < info.test_split < 1, 'cant split: {}'.format(info.val_split)
# 		*trainsets, testset = simple_split_dataset(dataset, 1-info.test_split, shuffle=True)
#
# 	if 'val_split' in info: # use/create validation set
# 		assert 0 < info.val_split < 1, 'cant split: {}'.format(info.val_split)
# 		trainsets = simple_split_dataset(trainsets[0], 1-info.val_split, shuffle=True)
#
# 	if 'repeat' in info and info.repeat > 0:
#
# 		trainset, *other = trainsets
#
# 		factor = info.repeat
#
# 		if factor > 1:
# 			trainset = Repeat_Dataset(trainset, factor)
#
# 		trainsets = trainset, *other
#
#
# 	if 'repeat_until' in info and info.repeat_until > 0:
#
# 		trainset, *other = trainsets
#
# 		N = len(trainset)
#
# 		factor = max(info.repeat_until // N, 1)
#
# 		if factor > 1:
# 			trainset = Repeat_Dataset(trainset, factor)
#
# 		trainsets = trainset, *other
#
#
# 	if 'repeat_until_batches' in info and info.repeat_until_batches > 0:
#
# 		trainset, *other = trainsets
#
# 		N = len(trainset)/info.batch_size
#
# 		factor = max(int(info.repeat_until_batches / N), 1)
#
# 		if factor > 1:
# 			trainset = Repeat_Dataset(trainset, factor)
#
# 		trainsets = trainset, *other
#
# 	return (*trainsets, testset) # testset is None if it doesnt exist or has to be loaded separately (with mode=='test')
#




