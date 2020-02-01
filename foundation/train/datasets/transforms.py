
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bisect import bisect_right

from ... import util
from ..registry import Component, Modifier, AutoModifier
from ..data import Dataset
from ...data import Device_Dataset, Info_Dataset, Testable_Dataset, Batchable_Dataset

@Dataset('concat')
class Concat(Info_Dataset):
	def __init__(self, A):
		datasets = A.pull('datasets')
		assert len(datasets), 'no datasets'
		super().__init__(datasets[0].din, datasets[0].dout)

		self.cumlens = np.cumsum([len(dataset) for dataset in datasets])
		self.datasets = datasets

	def __len__(self):
		return self.cumlens[-1]

	def __getitem__(self, item):
		idx = bisect_right(self.cumlens, item)
		if idx > 0:
			item -= int(self.cumlens[int(idx-1)])
		return self.datasets[idx][item]

	def pre_epoch(self, mode, epoch):
		for dataset in self.datasets:
			if isinstance(dataset, Info_Dataset):
				dataset.pre_epoch(mode, epoch)

	def post_epoch(self, mode, epoch, stats=None):
		for dataset in self.datasets:
			if isinstance(dataset, Info_Dataset):
				dataset.post_epoch(mode, epoch, stats=stats)


@AutoModifier('cropped')
class Cropped(Info_Dataset):
	'''
	Parent dataset must have a din that is an image

	'''

	def __init__(self, A, crop_size=None):

		if crop_size is None:
			crop_size = A.pull('crop_size')

		crop_loc = A.pull('crop_loc', 'center')
		# crop_key = A.pull('crop_key', None) # TODO

		if crop_loc is not 'center':
			raise NotImplementedError

		try:
			len(crop_size)
		except TypeError:
			crop_size = crop_size, crop_size
		assert len(crop_size) == 2, 'invalid cropping size: {}'.format(crop_size)

		if crop_size[0] != crop_size[1]:
			raise NotImplementedError('only square crops are implemented')

		assert hasattr(self, 'din'), 'This modifier requires a din (see Info_Dataset, eg. 3dshapes) ' \
		                                'in the dataset to be modified'

		assert len(self.din) == 3 or len(self.din) == 1, 'must be an image dataset'

		din = self.din

		A.din = (self.din[0], *crop_size)

		super().__init__(A)

		_, self.cy, self.cx = din
		self.cy, self.cx = self.cy // 2, self.cx // 2
		self.r = crop_size[0] // 2

	def __getitem__(self, item):
		sample = super().__getitem__(item)
		img, *other = sample

		img = img[..., self.cy-self.r:self.cy+self.r, self.cx-self.r:self.cx+self.r]

		return (img, *other)

@AutoModifier('interpolated')
class Interpolated(Info_Dataset):
	def __init__(self, A, interpolate_size=None, interpolate_mode=None):

		if interpolate_size is None:
			interpolate_size = A.pull('interpolate_size', None)

		if interpolate_mode:
			interpolate_mode = A.pull('interpolate_mode', 'bilinear')

		assert hasattr(self, 'din'), 'This modifier requires a din (see Info_Dataset, eg. 3dshapes) ' \
		                             'in the dataset to be modified'

		try:
			len(interpolate_size)
		except TypeError:
			interpolate_size = interpolate_size, interpolate_size
		assert len(interpolate_size) == 2, 'invalid cropping size: {}'.format(interpolate_size)

		assert len(self.din) == 3 or len(self.din) == 1, 'must be an image dataset'

		A.din = (self.din[0], *interpolate_size)

		super().__init__(A)

		self.interpolate_size = interpolate_size
		self.interpolate_mode = interpolate_mode

	def __getitem__(self, item):

		sample = self.__getitem__(item)
		img, *other = sample

		img = F.interpolate(img, self.interpolate_size, mode=self.interpolate_mode).squeeze(0)

		return (img, *other)






