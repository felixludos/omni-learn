
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from omnibelt import unspecified_argument

from bisect import bisect_right

from ... import util

from omnifig import AutoModifier, Modification

from ...data import register_dataset, Deviced, Dataset, Batchable, Subset_Dataset

@register_dataset('concat')
class Concat(Dataset):
	def __init__(self, A, din=None, dout=None, **kwargs):
		datasets = A.pull('datasets')
		assert len(datasets), 'no datasets'
		super().__init__(A, din=datasets[0].din, dout=datasets[0].dout, **kwargs)

		self.cumlens = np.cumsum([len(dataset) for dataset in datasets])
		self.datasets = datasets

	def __len__(self):
		return self.cumlens[-1]

	def __getitem__(self, item):
		idx = bisect_right(self.cumlens, item)
		return self.datasets[idx][item - int(self.cumlens[int(idx-1)]) if idx > 0 else item]

@AutoModifier('cropped')
class Cropped(Dataset):
	'''
	Parent dataset must have a din that is an image

	'''

	def __init__(self, A, crop_size=unspecified_argument, **kwargs):

		if crop_size is unspecified_argument:
			crop_size = A.pull('crop_size', None)

		crop_loc = A.pull('crop_loc', 'center')
		# crop_key = A.pull('crop_key', None) # TODO

		if crop_loc is not 'center':
			raise NotImplementedError

		if crop_size is not None:
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
			
			self.din = (self.din[0], *crop_size)

		super().__init__(A, **kwargs)
		
		self.r = None
		if crop_size is not None:
		
			_, self.cy, self.cx = din
			self.cy, self.cx = self.cy // 2, self.cx // 2
			self.r = crop_size[0] // 2


	def __getitem__(self, item):
		sample = super().__getitem__(item)
		if self.r is None:
			return sample
		
		img, *other = sample

		img = img[..., self.cy-self.r:self.cy+self.r, self.cx-self.r:self.cx+self.r]

		return (img, *other)



# @AutoModifier('interpolated')
class Interpolated(Dataset):
	def __init__(self, A, interpolate_size=None, interpolate_mode=None, **kwargs):

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

		raise NotImplementedError
		A.din = (self.din[0], *interpolate_size)

		super().__init__(A, **kwargs)

		self.interpolate_size = interpolate_size
		self.interpolate_mode = interpolate_mode

	def __getitem__(self, item):

		sample = self.__getitem__(item)
		img, *other = sample

		img = F.interpolate(img, self.interpolate_size, mode=self.interpolate_mode).squeeze(0)

		return (img, *other)


@AutoModifier('resamplable')
class Resamplable(Dataset):
	def __init__(self, A, budget=None):
		if budget is None:
			budget = A.pull('budget', None)

		super().__init__(A)

		self.budget = budget
		self.inds = self.resample(self.budget)

	def resample(self, budget=None):
		if budget is None:
			budget = self.budget
		inds = None
		if budget is not None:
			inds = torch.randint(0, super().__len__(), size=(budget,))
		return inds

	def __len__(self):
		return super().__len__() if self.budget is None else self.budget

	def pre_epoch(self, mode, epoch):
		if self.budget is not None and mode == 'train':
			self.inds = self.resample()
		super().pre_epoch(mode, epoch)

	def __getitem__(self, item):
		if self.budget is not None:
			item = self.inds[item]
		return super().__getitem__(item)



@Modification('blurred')
def blurred(dataset, config):

	if not isinstance(dataset, Deviced):
		raise NotImplementedError

	level = config.pull('blur-level', 5)
	assert level % 2 == 1, f'bad blur level: {level}'
	blur_type = config.pull('blur-type', 'uniform')

	assert blur_type == 'uniform', f'not implemented: {blur_type}'

	pad_type = config.pull('pad-type', 'reflect')
	padding = (level-1)//2

	C = dataset.din[0]

	blur = nn.Conv2d(C, 1, groups=C, bias=False, kernel_size=level,
	                      padding=padding, padding_mode=pad_type)
	blur.weight.requires_grad = False
	blur.weight.copy_(torch.ones_like(blur.weight)).div_(level**2)

	blur.to(dataset.device)

	key = config.pull('data_key', 'images')
	full = getattr(dataset, key)

	with torch.no_grad():
		full = blur(full).detach()

	setattr(dataset, key, full)

	return dataset


@Modification('subset')
def make_subset(dataset, info):
	num = info.pull('num', None)
	
	shuffle = info.pull('shuffle', True)
	
	if num is None or num == len(dataset):
		print('WARNING: no subset provided, using original dataset')
		return dataset
	
	assert num <= len(dataset), '{} vs {}'.format(num, len(dataset))
	
	inds = torch.randperm(len(dataset))[:num].numpy() if shuffle else np.arange(num)
	return Subset_Dataset(dataset, inds)



# @AutoModifier('blurred')
class Blurred(Dataset):

	def __init__(self, config):

		prepend = config.pull('prepend-tfm', False)
		if prepend:
			raise NotImplementedError

		level = config.pull('blur-level', 5)
		assert level % 2 == 1, f'bad blur level: {level}'
		blur_type = config.pull('blur-type', 'uniform')

		assert blur_type == 'uniform', f'not implemented: {blur_type}'

		pad_type = config.pull('pad-type', 'reflect')
		padding = (level-1)//2

		super().__init__(config)

		self.prepend = prepend

		C = self.din[0]

		self.pad_type = pad_type
		self.padding = padding

		self.blur = nn.Conv2d(C, 1, groups=C, bias=False, kernel_size=level,
		                      padding=padding, padding_mode=pad_type)
		self.blur.weight.requires_grad = False
		self.blur.weight.copy_(1).div_(level**2)

		self.done = False
		if isinstance(self, Deviced):

			self.blur.to(self.device)

			key = config.pull('data_key', 'images')
			full = getattr(self, key)

			with torch.no_grad():
				full = self.blur(full).detach()

			setattr(self, key, full)

			self.done = True



	def __getitem__(self, item):

		if self.done:
			return item

		raise NotImplementedError

		x, *other = item





		pass

