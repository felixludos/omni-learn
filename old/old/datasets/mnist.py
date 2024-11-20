
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
from omnibelt import unspecified_argument, agnosticproperty, agnostic
from omniplex import spaces, Dataset, Datastream

from ..novo.base import DataProduct, DataBuilder
from ..novo.base import hparam, inherit_hparams, submodule, submachine, material, space, indicator, machine, Structured

from .base import DownloadableDataset, ImageClassificationData



class _Torchvision_Toy_Dataset(DownloadableDataset, registry=ImageClassificationData):
	mode = hparam(None, inherit=True)
	download = hparam(False, inherit=True)
	_as_bytes = hparam(False)


	def __init_subclass__(cls, ident=None, **kwargs):
		if ident is None and cls._dirname is not None:
			ident = cls._dirname
		super().__init_subclass__(ident=ident, **kwargs)


	def _expected_size(self):
		return 10000 if self.mode == 'test' else 60000


	@material.from_indices('image')
	def get_observation(self, indices):
		images = self.images[indices]
		if self._as_bytes:
			return images
		return images.float().div(255)


	@material.from_indices('target')
	def get_target(self, indices):
		return self.targets[indices]
	@space('target')
	def target_space(self):
		return spaces.Categorical(10 if self._target_names is None else self._target_names)


	def _get_source_kwargs(self, root=unspecified_argument, train=unspecified_argument,
	                       download=unspecified_argument, **kwargs):
		if root is unspecified_argument:
			kwargs['root'] = self.root
		if train is unspecified_argument:
			kwargs['train'] = self.mode != 'test'
		if download is unspecified_argument:
			kwargs['download'] = self.download
		return kwargs


	_source_type = None
	_target_attr = 'targets'
	_target_names = None


	def download_data(self):
		self._create_source(download=True)


	def _create_source(self, **kwargs):
		src_kwargs = self._get_source_kwargs(**kwargs)
		src = self._source_type(**src_kwargs)
		return src


	def _prepare(self, *args, **kwargs):
		super()._prepare(*args, **kwargs)

		src = self._create_source()

		images = src.data
		if isinstance(images, np.ndarray):
			images = torch.as_tensor(images)
		if images.ndimension() == 3:
			images = images.unsqueeze(1)
		if images.size(1) not in {1,3}:
			images = images.permute(0,3,1,2)
		self.images = images

		targets = getattr(src, self._target_attr)
		if not isinstance(targets, torch.Tensor):
			targets = torch.as_tensor(targets)
		self.targets = targets



class _ResizableToyDataset(_Torchvision_Toy_Dataset):
	resize = hparam(True, inherit=True)


	@space('image')
	def observation_space(self):
		size = (32, 32) if self.resize else (28, 28)
		return spaces.Pixels(1, *size, as_bytes=self._as_bytes)


	def _prepare(self, *args, **kwargs):
		super()._prepare(*args, **kwargs)

		if self.resize:
			self.images = F.interpolate(self.images.float(), (32, 32), mode='bilinear').round().byte()



class MNIST(_ResizableToyDataset):
	_dirname = 'mnist'
	_source_type = torchvision.datasets.MNIST



class KMNIST(_ResizableToyDataset):
	_dirname = 'kmnist'
	_source_type = torchvision.datasets.KMNIST
	_target_names = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']



class FashionMNIST(_ResizableToyDataset):
	_dirname = 'fmnist'
	_source_type = torchvision.datasets.FashionMNIST
	_target_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']



class EMNIST(_ResizableToyDataset):
	_dirname = 'emnist'
	_source_type = torchvision.datasets.EMNIST
	_source_split_default_lens = {
		'balanced': {'train': 112800, 'test': 18800},
		'byclass': {'train': 697932, 'test': 116323},
		'bymerge': {'train': 697932, 'test': 116323},
		'digits': {'train': 240000, 'test': 40000},
		'letters': {'train': 124800, 'test': 20800},
		'mnist': {'train': 60000, 'test': 10000},
	}


	split = hparam('balanced', space=sorted(_source_split_default_lens.keys()))


	@space('target')
	def target_space(self):
		return spaces.Categorical(self._source_type.classes_split_dict[self.split])


	def _expected_size(self):
		return self._source_split_default_lens[self.split]['test' if self.mode == 'test' else 'train']


	def _get_source_kwargs(self, **kwargs):
		kwargs = super()._get_source_kwargs(**kwargs)
		kwargs['split'] = self.split
		return kwargs



class _Toy_RGB_Dataset(_Torchvision_Toy_Dataset):
	@space('image')
	def observation_space(self):
		return spaces.Pixels(3, 32, 32, as_bytes=self._as_bytes)



class SVHN(_Toy_RGB_Dataset):
	_dirname = 'svhn'
	_source_type = torchvision.datasets.SVHN
	_target_attr = 'labels'


	def _expected_size(self):
		return 26032 if self.mode == 'test' else 73257


	def _get_source_kwargs(self, **kwargs):
		kwargs = super()._get_source_kwargs(**kwargs)
		kwargs['split'] = 'train' if kwargs['train'] else 'test'
		del kwargs['train']
		return kwargs



class CIFAR_Data(ImageClassificationData, branch='cifar'):
	pass



class _CIFAR_Base(_Toy_RGB_Dataset, registry=CIFAR_Data):
	_dirname = 'cifar'



class CIFAR10(_CIFAR_Base, ident='10', is_default=True):
	name = 'cifar10'
	_source_type = torchvision.datasets.CIFAR10
	_target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



class CIFAR100(_CIFAR_Base, ident='100'):
	name = 'cifar100'
	_source_type = torchvision.datasets.CIFAR100
	_target_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
	                 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
	                 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
	                 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
	                 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
	                 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
	                 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
	                 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
	                 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
	                 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
	                 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
	                 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


