
import os

import numpy as np
import torch
from torch.nn import functional as F
import torchvision

from ..registry import create_component, Component
from ... import util
from ..data import Dataset

from ...data import Device_Dataset, Info_Dataset, Testable_Dataset, Batchable_Dataset

from .transforms import Interpolated

def _get_common_args(A):
	dataroot = A.pull('dataroot')

	download = A.pull('download', False)
	train = A.pull('train', True)

	return dataroot, {'download': download, 'train': train}


class Torchvision_Toy_Dataset(Device_Dataset, Testable_Dataset, Info_Dataset, Batchable_Dataset, util.Simple_Child):

	# def __init__(self, dataset=None, dataroot=None, download=True, label=True, label_attr='targets',
	#              train=True, din=(1,28,28), dout=10, resize=True,
	#              **kwargs):



	def __init__(self, dataset, train=True, label_attr=None, din=None, dout=None, **unused):
		'''
		Requires dataset object to wrap it (since this is a `util.Simple_Child`).

		Images must be in `dataset.data` and labels in `dataset.[label_attr]` (if provided).

		:param dataset: compatible pytorch torchvision dataset object
		:param train: bool
		:param label_attr: attr name used to access labels in `dataset`
		:param din: optional (if it has to be overwritten)
		:param dout: optional (if it has to be overwritten)
		'''

		# dataroot = A.pull('dataroot', None)
		# download = A.pull('download', True)
		#
		# label = A.pull('label', True)
		# label_attr = A.pull('label_attr', 'targets')
		#
		# train = A.pull('train', True)
		# din = A.pull('din', self.din)
		# dout = A.pull('dout', 10)

		if label_attr is None:
			dout = self.din if din is None else din

		super().__init__(din=din, dout=dout, train=train, _parent='dataset')

		self.dataset = dataset
		self.labeled = label_attr is not None

		images = self.dataset.data
		if isinstance(images, np.ndarray):
			images = torch.from_numpy(images)
		images = images.float().div(255)
		if images.ndimension() == 3:
			images = images.unsqueeze(1)
		self.labels = None
		if label_attr is not None:
			labels = getattr(self.dataset, label_attr)
			if not isinstance(labels, torch.Tensor):
				labels = torch.tensor(labels)
			self.register_buffer('labels', labels)

		self.register_buffer('images', images)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		img = self.images[item]
		if self.labeled:
			return img, self.labels[item]
		return img

@Dataset('mnist')
class MNIST(Torchvision_Toy_Dataset):
	din = (1, 28, 28)
	dout = 10

	def __init__(self, A):
		dataroot, kwargs = _get_common_args(A)
		dataset = torchvision.datasets.MNIST(os.path.join(dataroot, 'mnist'), **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, label_attr='targets' if labeled else None, **kwargs)

@Dataset('kmnist')
class KMNIST(Torchvision_Toy_Dataset):
	din = (1, 28, 28)
	dout = 10

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)
		dataset = torchvision.datasets.KMNIST(os.path.join(dataroot, 'kmnist'), **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, label_attr='targets' if labeled else None, **kwargs)

@Dataset('data/fmnist')
class FashionMNIST(Torchvision_Toy_Dataset):
	din = (1, 28, 28)
	dout = 10

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)
		dataset = torchvision.datasets.FashionMNIST(os.path.join(dataroot, 'fmnist'), **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, label_attr='targets' if labeled else None, **kwargs)

@Dataset('emnist')
class EMNIST(Torchvision_Toy_Dataset):
	din = (1, 28, 28)
	dout = 26

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)

		split = A.pull('split', 'letters')

		dataset = torchvision.datasets.EMNIST(os.path.join(dataroot, 'emnist'), split=split, **kwargs)

		dataset.targets -= 1  # targets use 1-based indexing :(

		labeled = A.pull('labeled', False)

		dout = None
		if labeled:
			if split != 'letters':
				raise NotImplementedError
			dout = 26

		super().__init__(dataset, label_attr='targets' if labeled else None, dout=dout, **kwargs)

@Dataset('svhn')
class SVHN(Torchvision_Toy_Dataset):
	din = (3, 32, 32)
	dout = 10

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)

		split = 'train' if kwargs['train'] else 'test'
		del kwargs['train']

		dataset = torchvision.datasets.SVHN(os.path.join(dataroot, 'svhn'), split=split, **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, label_attr='labels' if labeled else None, **kwargs)

@Dataset('cifar')
class CIFAR(Torchvision_Toy_Dataset):
	din = (3, 32, 32)
	dout = 10

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)

		classes = A.pull('classes', 10)

		assert classes in {10, 100}, 'invalid number of classes for cifar: {}'.format(classes)

		cls = torchvision.datasets.CIFAR10 if classes == 10 else torchvision.datasets.CIFAR100
		dataset = cls(os.path.join(dataroot, 'cifar'), **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, label_attr='targets' if labeled else None, **kwargs)

		self.images = self.images.permute(0, 3, 1, 2).contiguous()



