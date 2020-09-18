
from collections import deque
import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
# from .collectors import Batchable_Dataset, Device_Dataset # TODO: clean up import order

from .. import util

class Seedable(object):
	def set_seed(self, seed=None):
		return util.set_seed(seed)

class Featured_DataLoaderIter:
	
	def __init__(self, loader):
		super().__init__(loader)
		self.device = loader
		self.N = len(loader)
	
	def skip(self, num):
		for _ in range(num):
			self._next_index()
	
	def __next__(self):
		return util.to(super().__next__(), self.device)

class Featured_SingleProcessIter(Featured_DataLoaderIter, _SingleProcessDataLoaderIter):
	pass
class Featured_MultiProcessIter(Featured_DataLoaderIter, _MultiProcessingDataLoaderIter):
	pass

class Featured_DataLoader(Seedable, DataLoader):
	
	def __init__(self, *args, device=None, **kwargs):

		super().__init__(*args, **kwargs)

		if device is None:
			try:
				device = self.dataset.device
			except AttributeError:
				device = 'cpu'
		self.device = device
	
	# def __iter__(self):
	# 	itr = super().__iter__()
	#
	# 	def _skip(num):
	# 	itr.skip = _skip
	#
	# 	def _move_to():
	# 		batch = next(itr)
	# 		return util.to(batch, self.device)
	# 	itr.__next__ = _move_to
	#
	# 	return itr

	def get_dataset(self):
		return self.dataset

	def get_batch_size(self):
		return self.batch_size

	def __iter__(self):
		if self.num_workers == 0:
			return Featured_SingleProcessIter(self)
		else:
			return Featured_MultiProcessIter(self)



class BatchedDataLoader(Seedable): # loads full batches at a time (dataset must be Batched

	def __init__(self, dataset, batch_size, shuffle=True, drop_last=False,
	             auto_reset=False, device=None):
		self.dataset = dataset

		assert len(self.dataset) > batch_size, 'dataset is not large enough for a single batch: {} vs {}'.format(len(dataset), batch_size)

		# if not isinstance(dataset, Batchable_Dataset):
		# 	assert dataset.allow_batched(), 'this dataset doesnt seem to be compatible with a BatchedDataLoader: {}'.format(dataset)

		self.batch_size = batch_size
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.auto_reset = False
		self.num = len(self.dataset) // batch_size
		if self.num*batch_size != len(self.dataset) and not self.drop_last:
			self.num += 1

		if device is None:
			try:
				device = dataset.device
			except AttributeError:
				device = 'cpu'
		self.device = device
		# self.device = dataset.device if isinstance(dataset, Device_Dataset) else 'cpu'

	def __len__(self):
		return self.num

	def __iter__(self):
		return _BatchedDataLoaderIter(self.dataset, self.batch_size, self.shuffle, self.drop_last, self.auto_reset, self.device)

class _BatchedDataLoaderIter(object):
	def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, auto_reset=False, device=None):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.auto_reset = auto_reset
		self.device = device

		self.remainder = None
		self.batches = []
		self.idx = None
		self.reset()

	def reset(self):

		order = torch.randperm(len(self.dataset), device=self.device) if self.shuffle else torch.arange(0, len(self.dataset), device=self.device)

		history = None
		if self.remainder is not None:
			pull = self.batch_size - len(self.remainder)
			history = torch.cat([self.remainder, order[:pull]])
			order = order[pull:]

			self.remainder = None

		batches = list(torch.split(order, self.batch_size))

		if history is not None:
			batches.insert(0,history)

		last = batches[-1]
		if len(last) < self.batch_size:
			if self.drop_last or self.auto_reset:
				batches.pop()
				if self.auto_reset:
					self.remainder = last

		self.batches = batches
		self.idx = 0

	def skip(self, num):
		self.idx += num

	def __len__(self):
		return len(self.batches) - self.idx

	# def next_batch(self):
	# 	batch = next(self)
	# 	return util.to(batch, self.device)

	def __iter__(self):
		return self

	def __next__(self):
		if self.idx >= len(self.batches):
			if self.auto_reset:
				self.reset()
			else:
				raise StopIteration

		batch = self.dataset[self.batches[self.idx]]
		self.idx += 1

		return util.to(batch, self.device)
		return batch

	def __getitem__(self, item):
		return self.dataset[self.batches[item]]


def get_loaders(*datasets, batch_size=64, num_workers=0, shuffle=True, pin_memory=True,
		   drop_last=False, worker_init_fn=None, allow_batched=True, device='cpu'):

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

	loader_cls = Featured_DataLoader
	kwargs = {
		'batch_size': batch_size,
		'drop_last': drop_last,
		'device': device,
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


