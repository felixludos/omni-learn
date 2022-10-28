
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
from omnibelt import unspecified_argument

from omnidata import SupervisedDataset, DownloadableDataset, ImageDataset
from omnidata import spaces



class _Torchvision_Toy_Dataset(SupervisedDataset, ImageDataset, DownloadableDataset):
	def __init__(self, *, resize=True, as_bytes=False, mode=None, default_len=None,
	             observation_buffer=unspecified_argument, target_buffer=unspecified_argument, **kwargs):
		if default_len is None:
			default_len = 10000 if mode == 'test' else 60000

		if observation_buffer is unspecified_argument:
			observation_buffer = self.ImageBuffer(space=spaces.Pixels(1, 28, 28, as_bytes=as_bytes))
			if resize:
				observation_buffer.space.width = 32
				observation_buffer.space.height = 32

		if target_buffer is unspecified_argument:
			target_names = 10 if self._target_names is None else self._target_names
			target_buffer = self.Buffer(space=spaces.Categorical(target_names))
		super().__init__(mode=mode, default_len=default_len, **kwargs)

		if observation_buffer is not None:
			self.register_buffer('observation', observation_buffer)
		if target_buffer is not None:
			self.register_buffer('target', target_buffer)

		self.resize = resize


	def _get_source_kwargs(self, kwargs=None):
		if kwargs is None:
			kwargs = {}
		if 'root' not in kwargs:
			kwargs['root'] = self.get_root()
		kwargs['train'] = self.mode != 'test'
		if 'download' not in kwargs and self._auto_download is not None:
			kwargs['download'] = self._auto_download
		return kwargs

	_source_type = None
	_target_attr = 'targets'
	_target_names = None

	def _prepare(self, *args, **kwargs):
		src_kwargs = self._get_source_kwargs()
		src = self._source_type(**src_kwargs)

		images = src.data
		if isinstance(images, np.ndarray):
			images = torch.as_tensor(images)
		if images.ndimension() == 3:
			images = images.unsqueeze(1)
		if images.size(1) not in {1,3}:
			images = images.permute(0,3,1,2)
		if self.resize:
			images = F.interpolate(images.float(), (32, 32), mode='bilinear').round().byte()
		self.buffers['observation'].data = images

		targets = getattr(src, self._target_attr)
		if not isinstance(targets, torch.Tensor):
			targets = torch.as_tensor(targets)
		self.buffers['target'].data = targets

		super()._prepare(*args, **kwargs)



class MNIST(_Torchvision_Toy_Dataset):
	_name = 'mnist'
	_source_type = torchvision.datasets.MNIST



class KMNIST(_Torchvision_Toy_Dataset):
	_name = 'kmnist'
	_source_type = torchvision.datasets.KMNIST
	_target_names = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']



class FashionMNIST(_Torchvision_Toy_Dataset):
	_name = 'fmnist'
	_source_type = torchvision.datasets.FashionMNIST
	_target_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']



class EMNIST(_Torchvision_Toy_Dataset):
	_name = 'emnist'
	_source_type = torchvision.datasets.EMNIST
	_source_split_default_lens = {
		'byclass': {'train': 697932, 'test': 116323},
		'mnist': {'train': 60000, 'test': 10000},
		'balanced': {'train': 112800, 'test': 18800},
		'digits': {'train': 240000, 'test': 40000},
		'bymerge': {'train': 697932, 'test': 116323},
		'letters': {'train': 124800, 'test': 20800}
	}
	
	def __init__(self, split='letters', default_len=None, mode=None, target_buffer=unspecified_argument, **kwargs):
		if default_len is None and split in self._source_split_default_lens:
			default_len = self._source_split_default_lens['test' if mode == 'test' else 'train']

		if target_buffer is unspecified_argument:
			assert split in self._source_type.classes_split_dict, \
				f'{split} vs {list(self._source_type.classes_split_dict)}'
			target_buffer = self.Buffer(space=spaces.Categorical(self._source_type.classes_split_dict[split]))
		super().__init__(default_len=default_len, target_buffer=target_buffer, **kwargs)
		self._split = split


	def _get_source_kwargs(self, kwargs=None):
		kwargs = super()._get_source_kwargs()
		kwargs['split'] = self._split
		return kwargs



class _Toy_RGB_Dataset(_Torchvision_Toy_Dataset):
	def __init__(self, *, as_bytes=False, observation_buffer=unspecified_argument, **kwargs):
		if observation_buffer is unspecified_argument:
			observation_buffer = self.ImageBuffer(space=spaces.Pixels(3, 32, 32, as_bytes=as_bytes))
		super().__init__(observation_buffer=observation_buffer, resize=False, **kwargs)



class SVHN(_Toy_RGB_Dataset):
	_name = 'svhn'
	_source_type = torchvision.datasets.SVHN
	_target_attr = 'labels'

	def __init__(self, *, default_len=None, mode=None, **kwargs):
		if default_len is None:
			default_len = 26032 if mode == 'test' else 73257
		super().__init__(mode=mode, default_len=default_len, **kwargs)


	def _get_source_kwargs(self):
		kwargs = super()._get_source_kwargs()
		kwargs['split'] = 'train' if kwargs['train'] else 'test'
		del kwargs['train']
		return kwargs



class _CIFAR(_Toy_RGB_Dataset):
	def get_root(self, dataset_dir=None):
		if dataset_dir is None:
			dataset_dir = 'cifar'
		return super().get_root(dataset_dir=dataset_dir)



class CIFAR10(_CIFAR):
	_name = 'cifar10'
	_source_type = torchvision.datasets.CIFAR10
	_target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



class CIFAR100(_CIFAR):
	_name = 'cifar100'
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







