import os
from pathlib import Path
import pickle
import h5py as hf
import numpy as np
import torch
from omnibelt import unspecified_argument
from torch.nn import functional as F

from .. import util
from ...data import Dataset, Deviced, Batchable, Image_Dataset, DatasetBase

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

@Dataset('dsprites')
class dSprites(Deviced, Batchable, Image_Dataset):

	din = (1, 64, 64)
	dout = 5

	def __init__(self, A):

		dataroot = A.pull('dataroot', None)
		root = None

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
			root = os.path.join(dataroot, 'dsprites')
			path = os.path.join(root, filename)
			print('Loading dSprites dataset from disk: {}'.format(path))
			data = np.load(path, allow_pickle=True, encoding='bytes')

			self.meta = _rec_decode(data['metadata'][()])

			images = torch.from_numpy(data['imgs']).unsqueeze(1)

			self.register_buffer('images', images)

			if label_type is not None:
				if label_type == 'value':
					labels = torch.from_numpy(data['latents_values'][:,1:]).float()
				else:
					labels = torch.from_numpy(data['latents_classes'][:,1:]).int()
				self.register_buffer('labels', labels)

		self.labeled = hasattr(self, 'labels')
		self.root = root

	def get_raw_data(self):
		if self.labeled:
			return self.images, self.labels
		return self.images

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		imgs = self.images[item].float()
		if self.labeled:
			return imgs, self.labels[item]
		return imgs,

@Dataset('3dshapes')
class Shapes3D(Deviced, Batchable, Image_Dataset):

	din = (3, 64, 64)
	dout = 6

	def __init__(self, A, mode=None, labeled=None, **kwargs):

		root = None

		load_memory = A.pull('load_memory', True)
		if mode is None:
			mode = A.pull('mode', 'full')
		if labeled is None:
			labeled = A.pull('labeled', False)
		label_type = A.pull('label_type', 'class')
		noise = A.pull('noise', None)

		din = A.pull('din', self.din)
		dout = A.pull('dout', self.dout if labeled else din)

		if not load_memory:
			raise NotImplementedError

		super().__init__(A, din=din, dout=dout, **kwargs)
		
		dataroot = self.root
		
		if dataroot is None:
			raise NotImplementedError

		if dataroot is not None: # TODO: automate the downloading and formatting of the dataset (including split)
			if mode == 'full':
				file_name = '3dshapes.h5'
				print('WARNING: using full dataset (train+test)')
			elif mode == 'test':
				file_name = '3dshapes_test.h5'
			else:
				file_name = '3dshapes_train.h5'

			root = dataroot / '3dshapes'
			with hf.File(str(root / file_name), 'r') as data:

				images = data['images']
				images = torch.from_numpy(images[()]).permute(0,3,1,2)#.float().div(255)

				self.register_buffer('images', images)

				labels = data['labels']
				labels = torch.from_numpy(labels[()]).float()

				self.register_buffer('labels', labels)

		self.root = root

		# if noise is not None:
		# 	print('Adding {} noise'.format(noise))
		self.noise = noise

		self.labeled = labeled

		self.factor_order = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
		self.factor_sizes = [10, 10, 10, 8, 4, 15]
		self.factor_num_values = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                          'scale': 8, 'shape': 4, 'orientation': 15}
		
		if label_type == 'class':
			labels -= labels.min(0, keepdim=True)[0]
			labels *= ((torch.tensor(self.factor_sizes).float() - 1) / labels.max(0, keepdim=True)[0])
			self.labels = labels.long()

	def get_factor_sizes(self):
		return self.factor_sizes
	def get_factor_order(self):
		return self.factor_order

	def get_labels(self):
		return self.labels

	def get_raw_data(self):
		if self.labeled:
			return self.images, self.labels
		return self.images

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
class FullCelebA(Image_Dataset): # TODO: automate downloading and formatting

	din = (3, 218, 178)

	ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
	              'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
	              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
	              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
	              'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
	              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
	              'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
	              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',]

	def __init__(self, A, resize=unspecified_argument, label_type=unspecified_argument, mode=None, **kwargs):

		dataroot = A.pull('dataroot') # force to load data here.

		if label_type is unspecified_argument:
			label_type = A.pull('label_type', None)

		if mode is None:
			mode = A.pull('mode', 'train')
		if resize is unspecified_argument:
			resize = A.pull('resize', (256, 256))

		din = A.pull('din', self.din)

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

		dout = A.pull('dout', dout)

		_labels = {
			'attr': 'attrs',
			'identity': 'identities',
			'landmark': 'landmarks',
		}

		if resize is not None: # TODO: use Interpolated as modifier
			din = 3, *resize

		super().__init__(A, din=din, dout=dout, **kwargs)
		self.root = Path(dataroot) / 'celeba'
		name = 'celeba_test.h5' if mode == 'test' else 'celeba_train.h5'

		with hf.File(os.path.join(dataroot, 'celeba', name), 'r') as f:
			self.images = f['images'][()] # encoded as str
			self.labels = f[_labels[label_type]][()] if label_type is not None else None

			self.attr_names = f.attrs['attr_names']
			self.landmark_names = f.attrs['landmark_names']

		self.resize = resize

	def get_factor_sizes(self):
		return [2]*len(self.ATTRIBUTES)
	def get_factor_order(self):
		return self.ATTRIBUTES
	
	def get_attribute_key(self, idx):
		try:
			return self.ATTRIBUTES[idx]
		except IndexError:
			pass

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
	def __init__(self, A, resize=None, crop_size=128, **kwargs):
		# raise NotImplementedError('doesnt work since FullCelebA automatically resizes to (256,256)')
		crop_size = A.pull('crop_size', crop_size) # essentially sets this as the default

		super().__init__(A, resize=resize, crop_size=crop_size, **kwargs)


@Dataset('mpi3d')
class MPI3D(Deviced, Batchable, Image_Dataset):

	din = (3, 64, 64)
	dout = 7

	def __init__(self, A, mode=None, fid_ident=None, **kwargs):

		dataroot = A.pull('dataroot', None)

		if mode is None:
			mode = A.pull('mode', 'train')
		labeled = A.pull('labeled', False)

		din = A.pull('din', self.din)
		dout = A.pull('dout', self.dout if labeled else din)

		cat = A.pull('category', 'toy')

		assert cat in {'toy', 'sim', 'realistic', 'real', 'complex'}, 'invalid category: {}'.format(cat)
		if cat == 'sim':
			cat = 'realistic'

		super().__init__(A, din=din, dout=dout, fid_ident=cat, **kwargs)
		
		myroot = os.path.join(dataroot, 'mpi3d')
		self.root = Path(myroot)
		# fid_name = f'mpi3d_{cat}_stats_fid.pkl'
		# if fid_name in os.listdir(myroot):
		#
		# 	p = pickle.load(open(os.path.join(myroot, fid_name), 'rb'))
		#
		# 	self.fid_stats = p['m'], p['sigma']
		#
		# 	print('Found FID Stats')
		# else:
		# 	print('WARNING: Unable to load FID stats for this dataset')

		self.factor_order = ['object_color', 'object_shape', 'object_size', 'camera_height', 'background_color',
		                     'horizonal_axis', 'vertical_axis']
		self.factor_sizes = [6,6,2,3,3,40,40]

		self.factor_values = {
			'object_color': ['white', 'green', 'red', 'blue', 'brown', 'olive'],
			'object_shape': ['cone', 'cube', 'cylinder', 'hexagonal', 'pyramid', 'sphere'],
			'object_size': ['small', 'large'],
			'camera_height': ['top', 'center', 'bottom'],
			'background_color': ['purple', 'sea_green', 'salmon'],
			'horizonal_axis': list(range(40)),
			'vertical_axis': list(range(40)),
		}
		
		if cat == 'complex':
		
			self.factor_sizes[0], self.factor_sizes[1] = 4, 4
			self.factor_values['object_shape'] = ['mug', 'ball', 'banana', 'cup']
			self.factor_values['object_color'] = ['yellow', 'green', 'olive', 'red']

		sizes = np.array(self.factor_sizes)

		flr = np.cumprod(sizes[::-1])[::-1]
		flr[:-1] = flr[1:]
		flr[-1] = 1

		self._sizes = torch.from_numpy(sizes.copy()).long()
		self._flr = torch.from_numpy(flr.copy()).long()

		self.labeled = labeled

		fname = f'mpi3d_{cat}_{mode}.h5'
		if mode is None:
			fname = 'mpi3d_{}.npz'.format(cat)
			print('WARNING: using full dataset (train+test)')
			images = np.load(os.path.join(dataroot, 'mpi3d', fname))['images']
			indices = np.arange(len(images))
		else:
			path = os.path.join(dataroot, 'mpi3d', fname)
			if not os.path.isfile(path):
				path = os.path.join(dataroot, 'mpi3d', f'mpi3d_{cat}_train.h5')
				print(f'{fname} not found, loading training set instead')
			with hf.File(path, 'r') as f:
				images = f['images'][()]
				indices = f['indices'][()]

		self.register_buffer('images', torch.from_numpy(images).permute(0,3,1,2))
		self.register_buffer('indices', torch.from_numpy(indices))

	def get_factor_sizes(self):
		return self.factor_sizes
	def get_factor_order(self):
		return self.factor_order

	def get_labels(self):
		return self.get_label(self.indices)

	def get_label(self, inds):
		try:
			len(inds)
			inds = inds.view(-1,1)
		except TypeError:
			pass

		lvls = inds // self._flr
		labels = lvls % self._sizes

		return labels

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		imgs = self.images[idx].float().div(255)
		if self.labeled:
			labels = self.get_label(self.indices[idx])
			return imgs, labels
		return imgs,

