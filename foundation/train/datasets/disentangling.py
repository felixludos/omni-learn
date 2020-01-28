import os

import h5py as hf
import numpy as np
import torch
from torch.nn import functional as F

from ... import util
from ..registry import Component
from ..data import Dataset
from ...data import Device_Dataset, Info_Dataset, Testable_Dataset, Batchable_Dataset

from .transforms import Cropped

def _rec_decode(obj):
	'''
	recursively convert bytes to str
	:param obj: root obj
	:return:
	'''
	if isinstance(obj, dict):
		return {_rec_decode(k):_rec_decode(v) for k,v in obj.items()}
	if isinstance(obj, list):
		return [_rec_decode(x) for x in obj]
	if isinstance(obj, tuple):
		return tuple(_rec_decode(x) for x in obj)
	if isinstance(obj, bytes):
		return obj.decode()
	return obj

@Dataset('dSprites')
class dSprites(Device_Dataset, Info_Dataset, Batchable_Dataset):

	din = (1, 64, 64)
	dout = 5

	def __init__(self, A):

		dataroot = A.pull('dataroot', None)

		label_type = A.pull('label_type', None)

		din = A.pull('din', self.din)
		dout = A.pull('dout', din if label_type is None else self.dout)

		filename = A.pull('filename', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

		assert label_type in {None, 'value', 'class'}, 'Unknown type of label: {}'.format(label_type)
		# assert images is not None or dataroot is not None, 'nothing to use/load'

		if dout is None and label_type is not None:
			dout = 5 if label_type == 'value' else 113

		super().__init__(din=din, dout=din if dout is None else dout)

		if dataroot is not None:

			path = os.path.join(dataroot, 'dsprites', filename)
			print('Loading dSprites dataset from disk: {}'.format(path))
			data = np.load(path, allow_pickle=True, encoding='bytes')

			self.meta = _rec_decode(data['metadata'][()])

			images = torch.from_numpy(data['imgs']).unsqueeze(1)

			self.register_buffer('images', images)

			if label_type is not None:
				if label_type == 'values':
					labels = torch.from_numpy(data['latents_values'][:,1:]).float()
				else:
					labels = torch.from_numpy(data['latents_classes'][:,1:]).int()
				self.register_buffer('labels', labels)

		self.labeled = hasattr(self, 'labels')

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		imgs = self.images[item].float()
		if self.labeled:
			return imgs, self.labels[item]
		return imgs,

@Dataset('3dshapes')
class Shapes3D(Info_Dataset, Device_Dataset, Batchable_Dataset, Testable_Dataset):

	din = (3, 64, 64)
	dout = 6

	def __init__(self, A):

		dataroot = A.pull('dataroot', None)

		load_memory = A.pull('load_memory', True)
		train = A.pull('train', True)
		labeled = A.pull('labeled', False)
		noise = A.pull('noise', None)

		din = A.pull('din', self.din)
		dout = A.pull('dout', self.dout if labeled else din)

		if not load_memory:
			raise NotImplementedError

		super().__init__(din=din, dout=dout, train=train)

		if dataroot is not None: # TODO: automate the downloading and formatting of the dataset (including split)
			if train is None:
				file_name = '3dshapes.h5'
				print('WARNING: using full dataset (train+test)')
			elif train:
				file_name = '3dshapes_train.h5'
			else:
				file_name = '3dshapes_test.h5'
			with hf.File(os.path.join(dataroot, '3dshapes', file_name), 'r') as data:

				images = data['images']
				images = torch.from_numpy(images[()]).permute(0,3,1,2)#.float().div(255)

				self.register_buffer('images', images)

				if labeled:
					labels = data['labels']
					labels = torch.from_numpy(labels[()]).float()

					self.register_buffer('labels', labels)


		# if noise is not None:
		# 	print('Adding {} noise'.format(noise))
		self.noise = noise

		self.labeled = labeled

		self.factor_order = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
		self.factor_num_values = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                          'scale': 8, 'shape': 4, 'orientation': 15}

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		# if isinstance(item, (np.ndarray, torch.Tensor)):
		# 	item = item.tolist()
		# images = torch.from_numpy(self.images[item]).permute(2,0,1).float().div(255)
		images = self.images[item].float().div(255)
		if self.noise is not None:
			images = images.add(torch.randn_like(images).mul(self.noise)).clamp(0,1)
		if self.labeled:
			labels = self.labels[item]
			# labels = torch.from_numpy(self.labels[item]).float()
			return images, labels
		return images,


@Dataset('full-celeba') # probably shouldnt be used
class FullCelebA(Testable_Dataset, Info_Dataset): # TODO: automate downloading and formatting

	din = (3, 218, 178)

	def __init__(self, A):

		dataroot = A.pull('dataroot') # force to load data here.

		label_type = A.pull('label_type', None)

		train = A.pull('train', True)
		resize = A.pull('resize', (256, 256))

		din = A.pull('din', self.din)
		dout = A.pull('dout', None)

		_labels = {
			'attr': 'attrs',
			'identity': 'identities',
			'landmark': 'landmarks',
		}

		if resize is not None: # TODO: use Interpolated as modifier
			din = 3, *resize

		if dout is None:
			if label_type is None:
				dout = din
			elif label_type == 'attr':
				dout = 40
			elif label_type == 'landmark':
				dout = 10
			elif label_type == 'identity':
				dout = 1
			else:
				raise Exception('unknown {}'.format(label_type))

		super().__init__(din=din, dout=dout, train=train,)

		name = 'celeba_train.h5' if train else 'celeba_test.h5'

		with hf.File(os.path.join(dataroot, 'celeba', name), 'r') as f:
			self.images = f['images'][()] # encoded as str
			self.labels = f[_labels[label_type]][()] if label_type is not None else None

			self.attr_names = f.attrs['attr_names']
			self.landmark_names = f.attrs['landmark_names']

		self.resize = resize

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):

		img = torch.from_numpy(util.str_to_jpeg(self.images[item])).permute(2,0,1).float().div(255)

		if self.resize is not None:
			img = F.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear').squeeze(0)

		if self.labels is None:
			return img,

		lbl = torch.from_numpy(self.labels[item])

		return img, lbl


@Dataset('celeba')
class CelebA(Cropped, FullCelebA):
	def __init__(self, A):
		raise NotImplementedError('doesnt work since FullCelebA automatically resizes to (256,256)')
		crop_size = A.pull('crop_size', 128) # essentially sets this as the default
		super().__init__(A, crop_size=crop_size)


