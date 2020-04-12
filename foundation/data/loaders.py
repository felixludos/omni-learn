
from collections import deque
import torch

from torch.utils.data import DataLoader
from .collectors import Batchable_Dataset, Device_Dataset


class BatchedDataLoader(object): # loads full batches at a time (dataset must be Batched

	def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, auto_reset=False):
		self.dataset = dataset

		assert len(self.dataset) > batch_size, 'dataset is not large enough for a single batch: {} vs {}'.format(len(dataset), batch_size)

		if not isinstance(dataset, Batchable_Dataset):
			assert dataset.allow_batched(), 'this dataset doesnt seem to be compatible with a BatchedDataLoader: {}'.format(dataset)

		self.batch_size = batch_size
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.auto_reset = False
		self.num = len(self.dataset) // batch_size
		if self.num*batch_size != len(self.dataset) and not self.drop_last:
			self.num += 1


		self.device = dataset.device if isinstance(dataset, Device_Dataset) else 'cpu'

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


	def __next__(self):
		if self.idx >= len(self.batches):
			if self.auto_reset:
				self.reset()
			else:
				raise StopIteration

		batch = self.dataset[self.batches[self.idx]]
		self.idx += 1

		return batch

	def __getitem__(self, item):
		return self.dataset[self.batches[item]]

