
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
from omnibelt import unspecified_argument, agnosticproperty, agnostic
from omnidata import hparam, module, material, inherit_hparams, with_hparams
from omnidata import spaces, flavors

from ..novo.base import DataProduct, DataBuilder


class ToyData(DataBuilder, ident='toy', as_branch=True):
	pass


class CIFAR(ToyData, ident='cifar', as_branch=True):
	pass



class _Torchvision_Toy_Dataset(flavors.DownloadableRouter, flavors.SupervisedDataset, DataProduct, registry=ToyData):
	ImageBuffer = flavors.ImageBuffer

	resize = hparam(True)
	mode = hparam(None)
	_as_bytes = hparam(False)

	@hparam(hidden=True)
	def default_len(self):
		return 10000 if self.mode == 'test' else 60000

	@hparam(hidden=True)
	def observation_space(self):
		size = (32, 32) if self.resize else (28, 28)
		return spaces.Pixels(1, *size, as_bytes=self._as_bytes)

	@hparam(hidden=True)
	def target_space(self):
		return spaces.Categorical(10 if self._target_names is None else self._target_names)


	@material
	def observation(self):
		size = (32, 32) if self.resize else (28, 28)
		return self.ImageBuffer(space=spaces.Pixels(1, *size, as_bytes=self._as_bytes))


	def __init__(self, *, resize=True, as_bytes=False, mode=None, default_len=None,
	             observation_buffer=unspecified_argument, target_buffer=unspecified_argument, **kwargs):
		if default_len is None:
			default_len = 10000 if mode == 'test' else 60000

		if observation_buffer is unspecified_argument:
			observation_buffer = self.ImageBuffer(space=spaces.Pixels(1, 28, 28, as_bytes=as_bytes))
			if resize:
				observation_buffer.space.width = 32
				observation_buffer.space.height = 32
		elif resize:
			raise ValueError('Cannot resize a custom observation buffer')
		if target_buffer is unspecified_argument:
			target_buffer = self.Buffer(space=spaces.Categorical(10 if self._target_names is None
			                                                     else self._target_names))

		super().__init__(default_len=default_len, **kwargs)

		if observation_buffer is not None:
			self.register_material('observation', observation_buffer)
		if target_buffer is not None:
			self.register_material('target', target_buffer)

		self._resize = resize
		self.mode = mode


	def _get_source_kwargs(self, root=unspecified_argument, train=unspecified_argument,
	                       download=unspecified_argument, **kwargs):
		if root is unspecified_argument:
			kwargs['root'] = self.root
		if train is unspecified_argument:
			kwargs['train'] = self.mode != 'test'
		if download is unspecified_argument:
			kwargs['download'] = self._auto_download
		return kwargs

	_source_type = None
	_target_attr = 'targets'
	_target_names = None

	@agnostic
	def is_downloaded(self):
		return True

	def download(self):
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
		if self._resize:
			images = F.interpolate(images.float(), (32, 32), mode='bilinear').round().byte()
		self.get_material('observation').data = images

		targets = getattr(src, self._target_attr)
		if not isinstance(targets, torch.Tensor):
			targets = torch.as_tensor(targets)
		self.get_material('target').data = targets



class MNIST(_Torchvision_Toy_Dataset, ident='mnist'):
	_dirname = 'mnist'
	_source_type = torchvision.datasets.MNIST



class KMNIST(_Torchvision_Toy_Dataset, ident='kmnist'):
	_dirname = 'kmnist'
	_source_type = torchvision.datasets.KMNIST
	_target_names = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']



class FashionMNIST(_Torchvision_Toy_Dataset, ident='fmnist'):
	_dirname = 'fmnist'
	_source_type = torchvision.datasets.FashionMNIST
	_target_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']



class EMNIST(_Torchvision_Toy_Dataset, ident='emnist'):
	_dirname = 'emnist'
	_source_type = torchvision.datasets.EMNIST
	_source_split_default_lens = {
		'balanced': {'train': 112800, 'test': 18800},
		'byclass': {'train': 697932, 'test': 116323},
		'bymerge': {'train': 697932, 'test': 116323},
		'digits': {'train': 240000, 'test': 40000},
		'letters': {'train': 124800, 'test': 20800}
		'mnist': {'train': 60000, 'test': 10000},
	}

	split = hparam('split', default='mnist', choices=sorted(_source_split_default_lens.keys()))
	
	def __init__(self, split='letters', default_len=None, mode=None, target_buffer=unspecified_argument, **kwargs):
		if default_len is None and split in self._source_split_default_lens:
			default_len = self._source_split_default_lens['test' if mode == 'test' else 'train']

		if target_buffer is unspecified_argument:
			assert split in self._source_type.classes_split_dict, \
				f'{split} vs {list(self._source_type.classes_split_dict)}'
			target_buffer = self.Buffer(space=spaces.Categorical(self._source_type.classes_split_dict[split]))
		super().__init__(default_len=default_len, target_buffer=target_buffer, **kwargs)

		if target_buffer is not None:
			self.register_material('target', target_buffer)

		self._split = split


	def _get_source_kwargs(self, **kwargs):
		kwargs = super()._get_source_kwargs(**kwargs)
		kwargs['split'] = self._split
		return kwargs



class _Toy_RGB_Dataset(_Torchvision_Toy_Dataset):
	def __init__(self, *, as_bytes=False, observation_buffer=unspecified_argument, **kwargs):
		if observation_buffer is unspecified_argument:
			observation_buffer = self.ImageBuffer(space=spaces.Pixels(3, 32, 32, as_bytes=as_bytes))
		super().__init__(observation_buffer=observation_buffer, resize=False, **kwargs)



class SVHN(_Toy_RGB_Dataset, ident='svhn'):
	_dirname = 'svhn'
	_source_type = torchvision.datasets.SVHN
	_target_attr = 'labels'

	def __init__(self, *, default_len=None, mode=None, **kwargs):
		if default_len is None:
			default_len = 26032 if mode == 'test' else 73257
		super().__init__(mode=mode, default_len=default_len, **kwargs)


	def _get_source_kwargs(self, **kwargs):
		kwargs = super()._get_source_kwargs(**kwargs)
		kwargs['split'] = 'train' if kwargs['train'] else 'test'
		del kwargs['train']
		return kwargs



class _CIFAR_Base(_Toy_RGB_Dataset, registry=CIFAR):
	_dirname = 'cifar'



class CIFAR10(_CIFAR_Base, ident='10'):
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







