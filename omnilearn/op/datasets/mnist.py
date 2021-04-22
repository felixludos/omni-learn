
import os

import h5py as hf

from omnibelt import unspecified_argument

import numpy as np
import torch
from torch.nn import functional as F
import torchvision

from ... import util
from ...data import register_dataset, Batchable, Deviced, Downloadable, ImageDataset


from .transforms import Interpolated





class Torchvision_Toy_Dataset(Batchable, Deviced, Downloadable, ImageDataset): # TODO: enable (pytorch) transforms
	available_modes = {'train', 'test'}

	_default_label_attr = 'targets'

	def __init__(self, A, dataroot=None, mode=None, labeled=None, label_attr=unspecified_argument,
	             din=None, dout=None,
	             dataset_kwargs=None, _req_kwargs=None, **unused):
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
		
		if dataroot is None:
			dataroot = self._get_dataroot(A)
			
		if mode is None:
			mode = A.pull('mode', 'train')
		
		if dataset_kwargs is None:
			dataset_kwargs = self._get_dataset_kwargs(A, mode=mode, **unused)
		if 'root' not in dataset_kwargs:
			dataset_kwargs['root'] = str(dataroot)

		resize = A.pull('resize', True)
		C, H, W = self.din
		if resize and (H != 32 or W != 32):
			self.din = C, 32, 32

		if label_attr is unspecified_argument:
			if labeled is None:
				labeled = A.pull('labeled', True)
			label_attr = A.pull('label_attr', self._default_label_attr) if labeled else None

		if label_attr is None:
			dout = self.din if din is None else din

		if _req_kwargs is None:
			_req_kwargs = dataset_kwargs
		else:
			_req_kwargs.update(dataset_kwargs)

		super().__init__(A, din=din, dout=dout, dataroot=dataroot,
		                 _req_kwargs=_req_kwargs,
		                 **unused)
		self.add_existing_modes('test')

		self.labeled = label_attr is not None

		# self.root = root

		images = self.data
		if isinstance(images, np.ndarray):
			images = torch.from_numpy(images)
		images = images.float().div(255)
		if images.ndimension() == 3:
			images = images.unsqueeze(1)
		
		if resize:
			images = F.interpolate(images, (32, 32), mode='bilinear')
		
		if label_attr is None:
			self.labels = None
		else:
			labels = getattr(self, label_attr)
			if not isinstance(labels, torch.Tensor):
				labels = torch.tensor(labels)
			self.register_buffer('labels', labels)
			# delattr(self, label_attr)

		self.register_buffer('images', images)
		# del self.data

	def download(cls, A, **kwargs):
		cls(A, download=True)

	def get_raw_data(self):
		if self.labeled:
			return self.images, self.labels
		return self.images

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		img = self.images[item]
		if self.labeled:
			return img, self.labels[item]
		return img
	
	@classmethod
	def _get_dataroot(cls, A=None, ident=None, silent=False):
		dataroot = util.get_data_dir(A, silent=silent)
		if ident is None:
			assert hasattr(cls, '_dataset_ident'), f'{cls.__name__} must be registered if no ident is provided'
			ident = cls._dataset_ident
		return dataroot / ident
	
	@classmethod
	def _get_dataset_kwargs(cls, A, mode=None, download=None, train=None, **unused):
		
		if mode is None:
			mode = A.pull('mode', 'train')
		
		if download is None:
			download = A.pull('download', False)
		
		if train is None:
			train = A.pull('train', mode != 'test')
		
		return {'download': download, 'train':train}



@register_dataset('mnist')
class MNIST(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.MNIST):
	din = (1, 28, 28)
	dout = 10

@register_dataset('kmnist')
class KMNIST(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.KMNIST):
	din = (1, 28, 28)
	dout = 10

@register_dataset('fmnist')
class FashionMNIST(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.FashionMNIST):
	din = (1, 28, 28)
	dout = 10

@register_dataset('emnist')
class EMNIST(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.EMNIST):
	din = (1, 28, 28)
	dout = 26
	
	_split_keys = {
		'byclass': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxy',
		'bymerge': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt',
		'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
		'digits': '0123456789',
		'mnist': '0123456789',
		'balanced': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt',
	}
	
	def __init__(self, A, **kwargs):
		
		# split = A.pull('group', 'letters')
		# if split != 'letters':
		# 	raise NotImplementedError
		
		# dataset = torchvision.datasets.EMNIST(str(dataroot), split=split, **kwargs)


		# labeled = A.pull('labeled', False)
		
		
		# labels_key = self._split_keys.get(split, None) if labeled else None
		# dout = len(labels_key) if labeled else None
		
		# TODO: Automodifier for selected_classes
		# selected_classes = A.pull('selected_classes', None)
		# if selected_classes is not None:
		# 	sel = None
		# 	lbls = dataset.targets.clone()
		# 	full_key = labels_key
		# 	labels_key = []
		#
		# 	for i, c in enumerate(selected_classes):
		#
		# 		s = lbls == c
		# 		if sel is None:
		# 			sel = s
		# 		else:
		# 			sel += s
		# 		dataset.targets[s] = i
		# 		if full_key is not None:
		# 			labels_key.append(full_key[c])
		#
		# 	dataset.targets = dataset.targets[sel]
		# 	dataset.data = dataset.data[sel]
		#
		# 	if labeled:
		# 		dout = len(selected_classes)

		super().__init__(A, **kwargs)
		
		self.images = self.images.permute(0,1,3,2)
		
		if self.split == 'letters' and self.labels is not None:
			self.labels -= 1  # targets use 1-based indexing :(
		
		self.labels_key = self._split_keys[self.split]
	
	@classmethod
	def _get_dataset_kwargs(cls, A, split=None, **other):
		kwargs = super()._get_dataset_kwargs(A, **other)
		if split is None:
			split = A.pull('group', 'letters')
		kwargs['split'] = split
		return kwargs


@register_dataset('svhn')
class SVHN(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.SVHN):
	din = (3, 32, 32)
	dout = 10
	
	_default_label_attr = 'labels'

	@classmethod
	def _get_dataset_kwargs(cls, A, **unused):
		kwargs = super()._get_dataset_kwargs(A, **unused)
		kwargs['split'] = 'train' if kwargs['train'] else 'test'
		del kwargs['train']
		return kwargs


@register_dataset('cifar10')
class CIFAR10(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.CIFAR10):
	din = (3, 32, 32)
	dout = 10

	def _get_dataroot(cls, *args, ident=None, **kwargs):
		return super()._get_dataroot(*args, ident='cifar', **kwargs)


@register_dataset('cifar100')
class CIFAR100(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.CIFAR100):
	din = (3, 32, 32)
	dout = 100

	def _get_dataroot(cls, *args, ident=None, **kwargs):
		return super()._get_dataroot(*args, ident='cifar', **kwargs)

