
import os

import h5py as hf

import numpy as np
import torch
from torch.nn import functional as F
import torchvision

# from foundation.old.train.registry import create_component, Component
from ... import util
from ..data import Dataset

from ...data import standard_split, Device_Dataset, Info_Dataset, Splitable_Dataset, Testable_Dataset, Batchable_Dataset, Image_Dataset

from .transforms import Interpolated

def _get_common_args(A):
	dataroot = A.pull('dataroot')

	download = A.pull('download', False)
	mode = A.pull('mode', 'train', silent=True)
	train = A.pull('train', mode != 'test')

	return dataroot, {'download': download, 'train': train}


class Torchvision_Toy_Dataset(Device_Dataset, Testable_Dataset, Image_Dataset,
                              Info_Dataset, Batchable_Dataset, util.Simple_Child):

	# def __init__(self, dataset=None, dataroot=None, download=True, label=True, label_attr='targets',
	#              train=True, din=(1,28,28), dout=10, resize=True,
	#              **kwargs):



	def __init__(self, dataset, train=True, label_attr=None, din=None, dout=None,
	             A=None, root=None, **unused):
		'''
		Requires dataset object to wrap it (since this is a `util.Simple_Child`).

		Images must be in `dataset.data` and labels in `dataset.[label_attr]` (if provided).

		:param dataset: compatible pytorch torchvision dataset object
		:param train: bool
		:param label_attr: attr name used to access labels in `dataset`
		:param din: optional (if it has to be overwritten)
		:param dout: optional (if it has to be overwritten)
		:param root: path to dataset files
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

		resize = A.pull('resize', True)
		C, H, W = self.din
		if resize and (H != 32 or W != 32):
			self.din = C, 32, 32

		if label_attr is None:
			dout = self.din if din is None else din


		super().__init__(din=din, dout=dout, root=root, train=train, _parent='dataset')

		self.dataset = dataset
		self.labeled = label_attr is not None

		# self.root = root

		images = self.dataset.data
		if isinstance(images, np.ndarray):
			images = torch.from_numpy(images)
		images = images.float().div(255)
		if images.ndimension() == 3:
			images = images.unsqueeze(1)
		
		if resize:
			images = F.interpolate(images, (32, 32), mode='bilinear')
		
		self.labels = None
		if label_attr is not None:
			labels = getattr(self.dataset, label_attr)
			if not isinstance(labels, torch.Tensor):
				labels = torch.tensor(labels)
			self.register_buffer('labels', labels)

		self.register_buffer('images', images)

	def get_raw_data(self):
		if self.labeled:
			return self.images, self.labels
		return self.images

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		img = self.images[item]
		if self.labeled:
			return img, self.labels[item]
		return img

	def split(self, A):
		'''
		Should split the dataset according to info.val_split, and
		probably support shuffled splitting depending oninfo.shuffle_split
		:param A: config
		:return: tuple of "training" datasets
		'''
		mode = A.pull('mode', 'train', silent=True)
		if mode != 'test':
			mode = 'train'

		datasets = {mode: self}

		if mode == 'test':
			return datasets
		return standard_split(datasets, A)

@Dataset('mnist')
class MNIST(Torchvision_Toy_Dataset):
	din = (1, 28, 28)
	dout = 10

	def __init__(self, A):
		dataroot, kwargs = _get_common_args(A)
		root = os.path.join(dataroot, 'mnist')
		dataset = torchvision.datasets.MNIST(root, **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, A=A, label_attr='targets' if labeled else None,
		                 root=root, **kwargs)

@Dataset('kmnist')
class KMNIST(Torchvision_Toy_Dataset):
	din = (1, 28, 28)
	dout = 10

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)
		root = os.path.join(dataroot, 'kmnist')
		dataset = torchvision.datasets.KMNIST(root, **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, A=A, label_attr='targets' if labeled else None,
		                 root=root, **kwargs)

@Dataset('fmnist')
class FashionMNIST(Torchvision_Toy_Dataset):
	din = (1, 28, 28)
	dout = 10

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)
		root = os.path.join(dataroot, 'fmnist')
		dataset = torchvision.datasets.FashionMNIST(root, **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, A=A, label_attr='targets' if labeled else None,
		                 root=root, **kwargs)

@Dataset('emnist')
class EMNIST(Torchvision_Toy_Dataset):
	din = (1, 28, 28)
	dout = 26

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)
		root = os.path.join(dataroot, 'emnist')

		split = A.pull('split', 'letters')

		dataset = torchvision.datasets.EMNIST(root, split=split, **kwargs)

		dataset.targets -= 1  # targets use 1-based indexing :(

		labeled = A.pull('labeled', False)

		dout = None
		if labeled:
			if split != 'letters':
				raise NotImplementedError
			dout = 26

		super().__init__(dataset, A=A, label_attr='targets' if labeled else None,
		                 dout=dout, root=root, **kwargs)

@Dataset('svhn')
class SVHN(Torchvision_Toy_Dataset):
	din = (3, 32, 32)
	dout = 10

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)
		root = os.path.join(dataroot, 'svhn')

		split = 'train' if kwargs['train'] else 'test'
		del kwargs['train']

		dataset = torchvision.datasets.SVHN(root, split=split, **kwargs)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, A=A, label_attr='labels' if labeled else None,
		                 root=root, **kwargs)

@Dataset('cifar')
class CIFAR(Torchvision_Toy_Dataset):
	din = (3, 32, 32)
	dout = 10

	def __init__(self, A):

		dataroot, kwargs = _get_common_args(A)
		root = os.path.join(dataroot, 'cifar')

		classes = A.pull('classes', 10)

		assert classes in {10, 100}, 'invalid number of classes for cifar: {}'.format(classes)

		cls = torchvision.datasets.CIFAR10 if classes == 10 else torchvision.datasets.CIFAR100
		dataset = cls(root, **kwargs)

		dataset.data = dataset.data.transpose(0,3,1,2)

		labeled = A.pull('labeled', False)

		super().__init__(dataset, A=A, label_attr='targets' if labeled else None,
		                 root=root, **kwargs)



