
import os

import h5py as hf

from omnibelt import unspecified_argument

import numpy as np
import torch
from torch.nn import functional as F
import torchvision

from ... import util
from ...data import register_dataset, Batchable, Deviced, Downloadable, ImageDataset, Supervised


# from .transforms import Interpolated





class Torchvision_Toy_Dataset(Downloadable, Batchable, ImageDataset, Supervised):
	# TODO: enable (pytorch) transforms
	_available_modes = {'train', 'test'}

	_default_label_attr = 'targets'

	def __init__(self, A, dataroot=None, mode=None, supervised=None, target_attr=unspecified_argument,
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

		if target_attr is unspecified_argument:
			if supervised is None:
				supervised = A.pull('supervised', True)
			target_attr = A.pull('target_attr', self._default_label_attr if supervised else None)
			supervised = target_attr is not None

		if target_attr is None:
			dout = self.din if din is None else din

		if _req_kwargs is None:
			_req_kwargs = dataset_kwargs
		else:
			_req_kwargs.update(dataset_kwargs)

		super().__init__(A, din=din, dout=dout, dataroot=dataroot, supervised=supervised,
		                 _req_kwargs=_req_kwargs, **unused)
		self.add_available_modes('test')

		# self.root = root

		images = self.data
		if isinstance(images, np.ndarray):
			images = torch.from_numpy(images)
		images = images.float().div(255).clamp(1e-7, 1-1e-7)
		if images.ndimension() == 3:
			images = images.unsqueeze(1)
		
		if resize:
			images = F.interpolate(images, (32, 32), mode='bilinear')
		
		if target_attr is not None:
			labels = getattr(self, target_attr)
			if not isinstance(labels, torch.Tensor):
				labels = torch.tensor(labels)
			self.register_buffer('targets', labels)
			# delattr(self, label_attr)

		self.register_buffer('observations', images)
		# del self.data

	# def _update_data(self, indices):
	# 	self._replace_observations(self.images[indices])
	# 	if self.labeled:
	# 		self._replace_labels(self.labels[indices])

	@property
	def raw_folder(self) -> str:
		return os.path.join(self.root, self.__class__.__name__.split('_')[-1], 'raw')

	@property
	def processed_folder(self) -> str:
		return os.path.join(self.root, self.__class__.__name__.split('_')[-1], 'processed')

	def download(cls, A, **kwargs):
		cls(A, download=True)

	# def get_raw_data(self):
	# 	if self.labeled:
	# 		return self.images, self.labels
	# 	return self.images

	# def __len__(self):
	# 	return len(self.images)

	# def __getitem__(self, item):
	# 	img = self.images[item]
	# 	if self.labeled:
	# 		return img, self.labels[item]
	# 	return img
	
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

	_full_label_space = util.CategoricalDim(10)
	_all_label_names = list(map(str, range(10)))



@register_dataset('kmnist')
class KMNIST(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.KMNIST):
	din = (1, 28, 28)
	dout = 10

	_full_label_space = util.CategoricalDim(10)
	_all_label_names = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']



@register_dataset('fmnist')
class FashionMNIST(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.FashionMNIST):
	din = (1, 28, 28)
	dout = 10

	_full_label_space = util.CategoricalDim(10)
	_all_label_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']



@register_dataset('emnist')
class EMNIST(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.EMNIST):
	din = (1, 28, 28)
	dout = 26

	_full_label_space = util.CategoricalDim(26)

	_split_keys = {
		'byclass': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxy',
		'bymerge': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt',
		'letters': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
		'digits': '0123456789',
		'mnist': '0123456789',
		'balanced': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt',
	}
	_all_label_names = list(_split_keys['letters'])
	
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		self.observations = self.observation.permute(0,1,3,2)
		
		if self.split == 'letters' and self.targets is not None:
			self.targets -= 1  # targets use 1-based indexing :(
		
		self.labels_key = self._split_keys[self.split]
		self.dout = len(self.labels_key)
		self._all_label_names = list(self.labels_key)
		self._full_label_space = util.CategoricalDim(self.dout)
		
		
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
	_full_label_space = util.CategoricalDim(10)
	_all_label_names = list(map(str, range(10)))
	
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
	_full_label_space = util.CategoricalDim(10)
	_all_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


	def _get_dataroot(cls, *args, ident=None, **kwargs):
		return super()._get_dataroot(*args, ident='cifar', **kwargs)



@register_dataset('cifar100')
class CIFAR100(Torchvision_Toy_Dataset, util.InitWall, torchvision.datasets.CIFAR100):
	din = (3, 32, 32)
	dout = 100
	_full_label_space = util.CategoricalDim(100)
	_all_label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
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


	def _get_dataroot(cls, *args, ident=None, **kwargs):
		return super()._get_dataroot(*args, ident='cifar', **kwargs)

