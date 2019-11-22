
import os

import numpy as np
import torch
import torchvision

from ... import util
from ..data import register_dataset

from ...data import Device_Dataset, Info_Dataset, Testable_Dataset, Batchable_Dataset

class MNIST(Device_Dataset, Testable_Dataset, Info_Dataset, Batchable_Dataset, util.Simple_Child):

	def __init__(self, dataset=None, dataroot=None, download=True, label=True, label_attr='targets',
	             train=True, din=(1,28,28), dout=10,
	             **kwargs):

		assert dataset is not None or dataroot is not None, 'no dataset to use/load'

		super().__init__(din=din, dout=dout if label else din, train=train, _parent='dataset')

		if dataset is None:
			dataset = torchvision.datasets.MNIST(os.path.join(dataroot, 'mnist'),
			                                     train=train, download=download, **kwargs)

		self.dataset = dataset
		self.labeled = label

		images = self.dataset.data
		if isinstance(images, np.ndarray):
			images = torch.from_numpy(images)
		images = images.float().div(255)
		if images.ndimension() == 3:
			images = images.unsqueeze(1)
		self.labels = None
		if label:
			labels = getattr(self.dataset, label_attr)
			if not isinstance(labels, torch.Tensor):
				labels = torch.tensor(labels)
			self.register_buffer('labels', labels)

		self.register_buffer('images', images)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		if self.labeled:
			return self.images[item], self.labels[item]
		return self.images[item]

register_dataset('mnist', MNIST)


class KMNIST(MNIST):

	def __init__(self, dataroot, train=True, download=True, label=True, **kwargs):
		dataset = torchvision.datasets.KMNIST(os.path.join(dataroot, 'kmnist'),
		                                      train=train, download=download, **kwargs)

		super().__init__(dataset=dataset, train=train, label=label)

register_dataset('kmnist', KMNIST)


class FashionMNIST(MNIST):

	def __init__(self, dataroot, train=True, download=True, label=True, **kwargs):
		dataset = torchvision.datasets.FashionMNIST(os.path.join(dataroot, 'fmnist'),
		                                            train=train, download=download, **kwargs)

		super().__init__(dataset=dataset, train=train, label=label)

register_dataset('fmnist', FashionMNIST)


class EMNIST(MNIST):

	def __init__(self, dataroot, train=True, download=True, label=True, split='letters', **kwargs):
		dataset = torchvision.datasets.EMNIST(os.path.join(dataroot, 'emnist'), split=split,
		                                      train=train, download=download, **kwargs)

		super().__init__(dataset=dataset, train=train, label=label, dout=26)

register_dataset('emnist', EMNIST)


class SVHN(MNIST):
	def __init__(self, dataroot, train=True, download=True, label=True, split='letters', **kwargs):
		dataset = torchvision.datasets.SVHN(os.path.join(dataroot, 'svhn'), split='train' if train else 'test',
		                                      download=download, **kwargs)

		super().__init__(dataset=dataset, train=train, label=label, din=(3,32,32), label_attr='labels')

register_dataset('svhn', SVHN)


class CIFAR(MNIST):
	def __init__(self, dataroot, train=True, download=True, label=True, classes=10, **kwargs):
		assert classes in {10, 100}, 'classes must be 10 or 100'
		cls = torchvision.datasets.CIFAR10 if classes == 10 else torchvision.datasets.CIFAR100
		dataset = cls(os.path.join(dataroot, 'cifar'), train=train,
		                                    download=download, **kwargs)

		super().__init__(dataset=dataset, train=train, label=label, din=(3, 32, 32), dout=classes)

		self.images = self.images.permute(0,3,1,2).contiguous()

register_dataset('cifar', CIFAR)


