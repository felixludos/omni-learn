import sys, os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
import torchvision

from ... import util
from ..load_data import register_dataset

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
	def __init__(self, dataroot=None, images=None, labels=None, label_type='value', din=(1,64,64), dout=None,
				 filename='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'):
		assert label_type in {None, 'value', 'class'}, 'Unknown type of label: {}'.format(label_type)
		assert images is not None or dataroot is not None, 'nothing to use/load'

		if dout is None and label_type is not None:
			dout = 5 if label_type == 'value' else 113

		super().__init__(din=din, dout=dout)

		if images is None:

			data = np.load(os.path.join(dataroot,filename),
						   allow_pickle=True, encoding='bytes')

			self.meta = _rec_decode(data['metadata'][()])

			images = torch.from_numpy(data['imgs']).float()

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
		if self.labeled:
			return self.images[item], self.labels[item]
		return self.images[item]


register_dataset('dsprites', dSprites)

class Shapes3D(Device_Dataset, Testable_Dataset, Info_Dataset, Batchable_Dataset):


	pass

# register_dataset('3dshapes', Shapes3D)


class CelebA(Device_Dataset, Testable_Dataset, Info_Dataset, Batchable_Dataset):
	pass

# register_dataset('celeba', CelebA)
