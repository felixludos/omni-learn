
import sys, os

import numpy as np
import torch
from torch import nn

import omnifig as fig

from ..data import get_loaders, Info_Dataset, Subset_Dataset, simple_split_dataset, DataLoader, BatchedDataLoader
from .. import util
from .. import framework as fm

dataset_registry = fig.Registry()


def register_dataset(name, dataset):
	cmpn_name = f'dataset/{name}'
	fig.Component(cmpn_name)(dataset)

	dataset_registry.new(name, cmpn_name)


def Dataset(name):
	def _reg_fn(fn):
		nonlocal name
		register_dataset(name, fn)
		return fn
	
	return _reg_fn


# FD_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# print(FD_PATH)


# def get_loaders(*datasets, batch_size=64, num_workers=0, shuffle=True, pin_memory=True,
# 		   drop_last=False, worker_init_fn=None, allow_batched=True):
#
# 	if shuffle == 'all':
# 		shuffles = [True]*3
# 	elif shuffle:
# 		shuffles = [True, False, False]
# 	else:
# 		shuffles = [False]*3
#
# 	for ds in datasets:
# 		if ds is not None:
# 			break
# 	if ds is None:
# 		return datasets if len(datasets) > 1 else None # all are None
#
# 	loader_cls = DataLoader
# 	kwargs = {
# 		'batch_size': batch_size,
# 		'drop_last': drop_last,
# 	}
#
# 	if allow_batched:
# 		try:
# 			assert ds.allow_batched()
# 		except (AttributeError, AssertionError):
# 			pass
# 		else:
# 			print('Using batched data loader')
# 			loader_cls = BatchedDataLoader
# 	else:
#
# 		try:
# 			assert ds.get_device() == 'cpu'
# 		except AttributeError:
# 			pass
# 		except AssertionError:
# 			pin_memory = False
#
# 		kwargs.update({
# 			'pin_memory': pin_memory,
# 			'worker_init_fn': worker_init_fn,
# 			'num_workers': num_workers,
# 		})
#
#
# 	loaders = [(loader_cls(ds, shuffle=s, **kwargs) if ds is not None else None)
# 	           for ds, s in zip(datasets, shuffles)]
#
# 	# if not silent: # TODO: deprecated!
# 	# 	trainloader = loaders[0]
# 	# 	testloader = None if len(loaders) < 2 else loaders[-1]
# 	# 	valloader = None if len(loaders) < 3 else loaders[1]
# 	#
# 	# 	print('traindata len={}, trainloader len={}'.format(len(datasets[0]), len(trainloader)))
# 	# 	if valloader is not None:
# 	# 		print('valdata len={}, valloader len={}'.format(len(datasets[1]), len(valloader)))
# 	# 	if testloader is not None:
# 	# 		print('testdata len={}, testloader len={}'.format(len(datasets[-1]), len(testloader)))
# 	# 	print('Batch size: {} samples'.format(batch_size))
#
# 	if len(loaders) == 1:
# 		return loaders[0]
# 	return loaders



@fig.Script('load_data')
def load_data(A, mode=None):
	if '_type' not in A:
		name = A.pull('name')
		
		if name not in dataset_registry:
			raise Exception(f'No datasets named "{name}" is registered')
		
		A.push('_type', dataset_registry[name])
	
	# dataroot = A.pull('dataset.dataroot', None)
	
	
	if 'dataroot' not in A.dataset:
		# assert 'FOUNDATION_DATA_DIR' in os.environ, 'Neither dataroot nor $FOUNDATION_DATA_DIR found'
		# A.push('dataroot', os.environ['FOUNDATION_DATA_DIR'])
		dataroot = util.DEFAULT_DATA_PATH
		if 'FOUNDATION_DATA_DIR' in os.environ:
			dataroot = os.environ['FOUNDATION_DATA_DIR']
		else:
			print(f'$FOUNDATION_DATA_DIR not set, using default: {dataroot}')
		A.push('dataset.dataroot', dataroot)
	
	use_default_dataroot = A.pull('use_default_dataroot', True)
	A.push('dataroot', os.environ['FOUNDATION_DATA_DIR'] if 'FOUNDATION_DATA_DIR' in os.environ
	                  else util.DEFAULT_DATA_PATH, overwrite=use_default_dataroot)
	
	mode_override = mode is not None
	if mode is None:
		mode = 'train'
	mode = A.push('mode', mode, overwrite=mode_override)
	A.push('train', mode == 'train')
	
	seed = A.pull('seed')
	util.set_seed(seed)
	
	dataset = fig.create_component(A)
	
	# TODO: all of the below should be done in the dataset contructor
	# region Move into dataset constructor
	device = A.pull('device', 'cpu')
	try:
		dataset.to(device)
		print('Dataset moved to {}'.format(device))
	except AttributeError:
		pass
	except RuntimeError:
		print('Not enough memory to move dataset to {}'.format(device))
	
	if not isinstance(dataset, Info_Dataset):
		print('WARNING: it is strongly recommended for all datasets to be subclasses '
		      'of foundation.data.collectors.Info_Dataset, this dataset is not.')
	
	try:
		A.push('din', dataset.din)
		A.push('dout', dataset.dout)
	except AttributeError as e:
		print('WARNING: Dataset does not have a "din" and "dout"')
		raise e
	
	# endregion
	
	if 'split' in A:
		try:
			return dataset.split(A)  # should check mode to to know to categorize
		except AttributeError as e:
			raise e
	
	datasets = {mode: dataset}
	
	if mode == 'train':
		datasets = standard_split(datasets, A)
	return datasets


def standard_split(datasets, A):
	
	test_split = A.pull('test_split', None)
	if test_split is not None:
		shuffle = A.pull('test_shuffle', '<>shuffle', True)
		datasets['train'], datasets['test'] = simple_split_dataset(datasets['train'], 1 - test_split, shuffle=shuffle)
	
	val_split = A.pull('val_split', None)
	if val_split is not None:
		shuffle = A.pull('val_shuffle', '<>shuffle', True)
		datasets['train'], datasets['val'] = simple_split_dataset(datasets['train'], 1 - val_split, shuffle=shuffle)
	else:
		datasets['val'] = None
	
	return datasets


@fig.Modification('subset')
def make_subset(dataset, info):
	num = info.pull('num', None)
	
	shuffle = info.pull('shuffle', True)
	
	if num is None or num == len(dataset):
		print('WARNING: no subset provided, using original dataset')
		return dataset
	
	assert num <= len(dataset), '{} vs {}'.format(num, len(dataset))
	
	inds = torch.randperm(len(dataset))[:num].numpy() if shuffle else np.arange(num)
	return Subset_Dataset(dataset, inds)


