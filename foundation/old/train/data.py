

import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from foundation import util
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from foundation.data.loaders import BatchedDataLoader
from foundation.data.collectors import *
from foundation.data.collate import _collate_movable
from .registry import create_component, Component, Modifier, Modification
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

	for ds in datasets:
		if ds is not None:
			break
	if ds is None:
		return datasets if len(datasets) > 1 else None # all are None

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


	loaders = [(loader_cls(ds, shuffle=s, **kwargs) if ds is not None else None)
	           for ds, s in zip(datasets, shuffles)]

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
	elif 'name' in info:
		del info.name

	if 'dataroot' not in info and 'FOUNDATION_DATA_DIR' in os.environ:
		info.dataroot = os.environ['FOUNDATION_DATA_DIR']

	if mode is not None:  # TODO: doesn't allow for other modes
		info.train = mode == 'train'
		info.mode = mode

	if created_config:
		info.begin()

	dataset = create_component(info)

	if created_config:
		info.abort()

	return dataset

class UntestableDatasetError(Exception):
	def __init__(self, name):
		super().__init__('Unable to only load testset for {}, since it doesnt subclass Testable_Dataset'.format(name))


def default_load_data(info, mode=None):
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

	if not isinstance(dataset, Info_Dataset):
		print('WARNING: it is strongly recommended for all datasets to be subclasses '
		      'of fd.data.collectors.Info_Dataset, this dataset is not.')
	# 	dataset.pre_epoch = lambda x, y: 0
	# 	dataset.post_epoch = lambda x, y, z: 0
	# 	raise NotImplementedError

	# if mode == 'test':
	# 	return dataset

	return dataset

@Modification('subset')
def make_subset(dataset, info):
	
	num = info.pull('num', None)
	
	shuffle = info.pull('shuffle', True)
	
	if num is None or num == len(dataset):
		print('WARNING: no subset provided, using original dataset')
		return dataset
	
	assert num <= len(dataset), '{} vs {}'.format(num, len(dataset))
	
	inds = torch.randperm(len(dataset))[:num].numpy() if shuffle else np.arange(num)
	return Subset_Dataset(dataset, inds)



