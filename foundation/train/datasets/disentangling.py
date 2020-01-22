import os

import h5py as hf
import numpy as np
import torch
from torch.nn import functional as F

from ... import util
from ..data import register_dataset
from ...data import Device_Dataset, Info_Dataset, Testable_Dataset, Batchable_Dataset

def _rec_decode(obj):
	if isinstance(obj, dict):
		return {_rec_decode(k):_rec_decode(v) for k,v in obj.items()}
	if isinstance(obj, list):
		return [_rec_decode(x) for x in obj]
	if isinstance(obj, tuple):
		return tuple(_rec_decode(x) for x in obj)
	if isinstance(obj, bytes):
		return obj.decode()
	return obj

class dSprites(Device_Dataset, Info_Dataset, Batchable_Dataset):
	def __init__(self, dataroot=None, images=None, labels=None, label_type=None, din=(1,64,64), dout=None,
				 filename='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'):
		assert label_type in {None, 'value', 'class'}, 'Unknown type of label: {}'.format(label_type)
		assert images is not None or dataroot is not None, 'nothing to use/load'

		if dout is None and label_type is not None:
			dout = 5 if label_type == 'value' else 113

		super().__init__(din=din, dout=din if labels is None else dout)

		if images is None:

			data = np.load(os.path.join(dataroot, 'dsprites', filename),
						   allow_pickle=True, encoding='bytes')

			self.meta = _rec_decode(data['metadata'][()])

			images = torch.from_numpy(data['imgs']).unsqueeze(1)

			if label_type is not None:
				if label_type == 'values':
					labels = torch.from_numpy(data['latents_values'][:,1:]).float()
				else:
					labels = torch.from_numpy(data['latents_classes'][:,1:]).int()

		self.register_buffer('images', images)
		self.labeled = labels is not None
		if self.labeled:
			self.register_buffer('labels', labels)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		imgs = self.images[item].float()
		if self.labeled:
			return imgs, self.labels[item]
		return imgs,


register_dataset('dsprites', dSprites)

class Shapes3D(Info_Dataset, Device_Dataset, Batchable_Dataset, Testable_Dataset):

	def __init__(self, dataroot=None, images=None, labels=None, din=(3, 64, 64), dout=None, load_memory=True,
	             train=True, noise=None):
		assert images is not None or dataroot is not None, 'nothing to use/load'

		if not load_memory:
			raise NotImplementedError

		super().__init__(din=din, dout=din if labels is None else dout, train=train)

		if images is None:

			with hf.File(os.path.join(dataroot, '3dshapes', '3dshapes.h5' if train is None
				else '3dshapes_{}.h5'.format('train' if self.train else 'test')), 'r') as data:

				images = data['images']
				images = torch.from_numpy(images[()]).permute(0,3,1,2)#.float().div(255)

				if labels is not None:
					labels = data['labels']
					labels = torch.from_numpy(labels[()]).float()

		if noise is not None:
			print('Adding {} noise'.format(noise))
		self.noise = noise

		self.register_buffer('images', images)
		self.labeled = labels is not None
		if self.labeled:
			self.register_buffer('labels', labels)

		self.factor_order = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
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

register_dataset('3dshapes', Shapes3D)


class CelebA(Testable_Dataset, Info_Dataset):


	def __init__(self, dataroot, label_type=None, train=True, resize=True):

		_labels = {
			'attr': 'attrs',
			'identity': 'identities',
			'landmark': 'landmarks',
		}

		din = (3, 218, 178)
		if resize:
			resize = (256, 256)
			din = (3, 256, 256)
		else:
			resize = None

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
			self.images = f['images'][()]
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

register_dataset('celeba', CelebA)
