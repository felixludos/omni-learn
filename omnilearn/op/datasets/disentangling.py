import sys, os
from pathlib import Path
import subprocess
import pickle
import zipfile
import h5py as hf
import numpy as np
import torch
from omnibelt import unspecified_argument, get_printer
import omnifig as fig
from torch.nn import functional as F

prt = get_printer(__name__)

try:
	from google.cloud import storage
except ImportError:
	prt.info('Failed to import gsutil (install with "pip install gsutil")')

try:
	import wget
except ImportError:
	prt.info('Failed to import wget (install with "pip install wget")')
	

from ... import util
from ...data import register_dataset, Deviced, Batchable, Splitable, ImageDataset, \
	Downloadable, Dataset, MissingDatasetError, Subset_Dataset

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


class DisentanglementDataset(ImageDataset):
	
	def get_factor_sizes(self):
		raise NotImplementedError
	def get_factor_order(self):
		raise NotImplementedError
	
	def get_observations(self):
		raise NotImplementedError
	def get_labels(self):
		raise NotImplementedError
	def update_data(self, indices):
		raise NotImplementedError

@fig.AutoModifier('selected')
class Selected(Splitable):
	def __init__(self, A, ordered=unspecified_argument, unordered=unspecified_argument,
	             accepted=unspecified_argument, invert=None, eval_name=None, **kwargs):
		
		if accepted is unspecified_argument:
			accepted = A.pull('accepted', {})
		if ordered is unspecified_argument:
			ordered = A.pull('ordered', None)
		if unordered is unspecified_argument:
			unordered = A.pull('unordered', None)
		
		if invert is None:
			invert = A.pull('invert', False)
		
		if eval_name is None:
			eval_name = A.pull('eval-name', 'extra')
		
		if accepted is not None or ordered is not None or unordered is not None:
			A.push('load-labels', True)
		
		self.accepted = accepted
		self.ordered = ordered
		self.unordered = unordered
		
		self.invert = invert
		
		self.eval_name = eval_name
		
		super().__init__(A, **kwargs)
		
		
	def _split_load(self, dataset):
		
		sizes = self.get_factor_sizes()
		flts = {}
		
		if self.accepted is not None:
			flts.update({int(k): v for k, v in self.accepted.items()})
		
		if self.ordered is not None:
			for idx, ratio in self.ordered.items():
				idx = int(idx)
				if idx not in flts and ratio != 0:
					N = sizes[idx]
					vals = np.arange(N)
					num = min(np.ceil(abs(ratio) * N), N - 1) if isinstance(ratio, float) \
						else max(1, min(abs(ratio), N - 1))
					vals = vals[:num] if ratio > 0 else vals[-num:]
					vals = vals.tolist()
					
					flts[idx] = vals
		
		if self.unordered is not None:
			raise NotImplementedError
		
		self.accepted = flts
		
		if len(flts):
			dataset = self._select(dataset, flts)
			
		return super()._split_load(dataset)
		
	def _select(self, dataset, flts):
		lbls = self.get_labels()
		ok = None
		for idx, valid in flts.items():
			valid = torch.tensor(valid, device=lbls.device).unsqueeze(0)
			vals = lbls[:, idx:idx+1]
			for v in valid:
				s = vals.sub(v).eq(0).sum(-1)
				if ok is None:
					ok = s
				else:
					ok *= s
		ok = ok.bool()
		if self.invert:
			ok = torch.logical_not(ok)
			
		inds = torch.arange(len(lbls), device=lbls.device)
		
		extra = Subset_Dataset(dataset, inds[torch.logical_not(ok)])
		self.register_mode(self.eval_name, extra)
		
		dataset = Subset_Dataset(dataset, inds[ok])
		self.register_mode(self.split_src, dataset)

		return dataset


@register_dataset('dsprites')
class dSprites(Deviced, Batchable, Downloadable, DisentanglementDataset):

	din = (1, 64, 64)
	dout = 5

	def __init__(self, A, **kwargs):

		label_type = A.pull('label_type', None)

		din = A.pull('din', self.din)
		dout = A.pull('dout', din if label_type is None else self.dout)

		filename = A.pull('filename', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

		assert label_type in {None, 'value', 'class'}, 'Unknown type of label: {}'.format(label_type)
		# assert images is not None or dataroot is not None, 'nothing to use/load'

		if dout is None and label_type is not None:
			dout = 5 if label_type == 'value' else 113

		super().__init__(A, din=din, dout=din if dout is None else dout, **kwargs)

		dataroot = self.root

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


@register_dataset('3dshapes')
class Shapes3D(Deviced, Batchable, Downloadable, DisentanglementDataset):

	din = (3, 64, 64)
	dout = 6

	def __init__(self, A, mode=None, labeled=None, load_labels=None, **kwargs):

		if mode is None:
			mode = A.pull('mode', 'full')
		if labeled is None:
			labeled = A.pull('labeled', False)
		if load_labels is None and not labeled:
			load_labels = A.pull('load-labels', False)
		load_labels = load_labels or labeled
		label_type = A.pull('label_type', 'class') if load_labels else None

		din = A.pull('din', self.din)
		dout = A.pull('dout', self.dout if labeled else din)

		super().__init__(A, din=din, dout=dout, **kwargs)
		self.add_existing_modes('test')
		
		self.factor_order = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
		self.factor_sizes = [10, 10, 10, 8, 4, 15]
		self.factor_num_values = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
		                          'scale': 8, 'shape': 4, 'orientation': 15}
		
		raw_mins = torch.tensor([  0.  ,   0.  ,   0.  ,   0.75,   0.  , -30.  ]).float()
		raw_maxs = torch.tensor([  0.9 ,  0.9 ,  0.9 ,  1.25,  3.  , 30.  ]).float()
		
		dataroot = self.root / '3dshapes'
		dataroot.mkdir(exist_ok=True)
		self.root = dataroot
		
		path = dataroot / '3dshapes.h5'
		if not path.is_file():
			download = A.pull('download', False)
			if download:
				self.download(A)
			else:
				raise MissingDatasetError('3dshapes')
		
		self.images, self.labels, self.indices = None, None, None
		with hf.File(path, 'r') as data:
			
			images = torch.from_numpy(data['images'][()])
			labels = torch.from_numpy(data['labels'][()]).float() if load_labels else None
			
			indices = None if mode == 'full' else \
				(data['test_idx'][()] if mode == 'test' else data['train_idx'][()])
			if indices is not None:
				indices = torch.from_numpy(indices).long()
				images = images[indices]
				if labels is not None:
					labels = labels[indices]
				self.register_buffer('indices', indices)
			else:
				prt.warning('using the full dataset (train+test)')
			
			images = images.permute(0, 3, 1, 2)#.float().div(255)
			self.register_buffer('images', images)
			
			if labels is not None:
				if label_type == 'class':
					nums = torch.tensor(self.factor_sizes).float() - 1
					labels -= raw_mins
					labels /= raw_maxs - raw_mins
					labels *= nums
					labels = labels.round().long()
				self.register_buffer('labels', labels)
				
		self.labeled = labeled
		self.label_type = label_type

		
	_source_url = 'gs://3d-shapes/3dshapes.h5'
	@classmethod
	def download(cls, A, dataroot=None, **kwargs):
		
		if dataroot is None:
			dataroot = util.get_data_dir(A) / '3dshapes'
		
		dest = dataroot / '3dshapes.h5'
		
		force_download = A.pull('force-download', False, silent=True)
		
		if not dest.is_file() or force_download:
			print(f'Downloading 3dshapes dataset to {str(dest)} ...', end='')
			subprocess.run(['gsutil', 'cp', cls._source_url, str(dest)])
			print(' done!')
		
		ratio = A.pull('separate-testset', 0.2)
		
		rng = np.random.RandomState(0)
		
		with hf.File(dest, 'r+') as f:
			if ratio is not None and ratio > 0 and 'train_idx' not in f:
				
				N, H, W, C = f['images'].shape
				
				test_N = int(N * ratio)
				
				order = rng.permutation(N)
				
				train_idx = order[:-test_N]
				train_idx.sort()
				test_idx = order[-test_N:]
				test_idx.sort()
				
				f.create_dataset('train_idx', data=train_idx)
				f.create_dataset('test_idx', data=test_idx)

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

	def update_data(self, indices):
		self.images = self.images[indices]
		if self.labeled:
			self.labels = self.labels[indices]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		# if isinstance(item, (np.ndarray, torch.Tensor)):
		# 	item = item.tolist()
		images = self.images[item].float().div(255)
		# if self.noise is not None:
		# 	images = images.add(torch.randn_like(images).mul(self.noise)).clamp(0,1)
		if self.labeled:
			labels = self.labels[item]
			# labels = torch.from_numpy(self.labels[item]).float()
			return images, labels
		return images,


@register_dataset('full-celeba')  # probably shouldnt be used
class FullCelebA(Downloadable, ImageDataset):  # TODO: automate downloading and formatting
	
	din = (3, 218, 178)
	
	ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
	              'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
	              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
	              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
	              'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
	              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
	              'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
	              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', ]
	
	def __init__(self, A, resize=unspecified_argument, label_type=unspecified_argument, mode=None, **kwargs):
		
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
		
		if resize is not None:  # TODO: use Interpolated as modifier
			din = 3, *resize
		
		super().__init__(A, din=din, dout=dout, **kwargs)
		# self.add_existing_modes('val', 'test')
		self.root = Path(self.root) / 'celeba'
		dataroot = self.root
		name = 'celeba_test.h5' if mode == 'test' else 'celeba_train.h5'
		
		with hf.File(dataroot/name, 'r') as f:
			self.images = f['images'][()]  # encoded as str
			self.labels = f[_labels[label_type]][()] if label_type is not None else None
			
			self.attr_names = f.attrs['attr_names']
			self.landmark_names = f.attrs['landmark_names']
		
		self.resize = resize
	
	# google drive ids
	_google_drive_ids = {
		'list_eval_partition.txt': '0B7EVK8r0v71pY0NSMzRuSXJEVkk',
		'identity_CelebA.txt': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
		'list_attr_celeba.txt': '0B7EVK8r0v71pblRyaVFSWGxPY0U',
		'list_bbox_celeba.txt': '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
		'list_landmarks_align_celeba.txt': '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
	}
	_google_drive_image_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
	
	@classmethod
	def download(cls, A, dataroot=None, **kwargs):
		
		raise NotImplementedError('Unfortunately, automatically downloading CelebA is current not supported.')
		
		dataroot = util.get_data_dir(A)
		dataroot = dataroot / 'celeba'
		dataroot.mkdir(exist_ok=True)
		
		prt.warning(
			'Downloading CelebA doesn\'t seem to work currently (issues with google drive), instead it must be downloaded manually:'
			'\n1. create a directory "celeba" in the local_data directory'
			'\n2. download "img_align_celeba.zip" into that directory'
			'\n3. download label files: {}'
			'\n4. extract "img_align_celeba.zip" to a directory "img_align_celeba"'.format(
				', '.join(f'"{c}"' for c in cls._google_drive_ids.keys())))
		
		raise NotImplementedError
		
		imgdir = dataroot / 'img_align_celeba'
		
		pbar = A.pull('pbar', None)
		
		util.download_file_from_google_drive('0B7EVK8r0v71pOXBhSUdJWU1MYUk', dataroot, pbar=pbar)
		
		for name, ID in cls._google_drive_ids.items():
			path = dataroot / name
			if not path.exists():
				util.download_file_from_google_drive(ID, dataroot, name, pbar=pbar)
		
		if not imgdir.is_dir():
			
			imgpath = dataroot / 'img_align_celeba.zip'
			
			# download zip
			if not imgpath.exists():
				imgpath = util.download_file_from_google_drive(cls._google_drive_image_id, dataroot, pbar=pbar)
			
			# extract zip
			imgdir.mkdir(exist_ok=True)
			with zipfile.ZipFile(str(imgpath), 'r') as zip_ref:
				zip_ref.extractall(str(imgdir))
			
			# os.remove(str(imgpath))
			
		raise NotImplementedError
	
	def get_factor_sizes(self):
		return [2] * len(self.ATTRIBUTES)
	
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
		
		img = torch.from_numpy(util.str_to_jpeg(self.images[item])).permute(2, 0, 1).float().div(255)
		
		if self.resize is not None:
			img = F.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear').squeeze(0)
		
		if self.labels is None:
			return img,
		
		lbl = torch.from_numpy(self.labels[item])
		
		return img, lbl


@register_dataset('celeba')
class CelebA(Cropped, FullCelebA):
	def __init__(self, A, resize=None, crop_size=128, **kwargs):
		# raise NotImplementedError('doesnt work since FullCelebA automatically resizes to (256,256)')
		crop_size = A.pull('crop_size', crop_size) # essentially sets this as the default

		super().__init__(A, resize=resize, crop_size=crop_size, **kwargs)


@register_dataset('mpi3d')
class MPI3D(Deviced, Batchable, Downloadable, DisentanglementDataset):

	din = (3, 64, 64)
	dout = 7

	def __init__(self, A, mode=None, fid_ident=None, **kwargs):

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
		self.add_existing_modes('test')
		dataroot = self.root / 'mpi3d'
		self.root = dataroot
		
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

		path = dataroot / f'mpi3d_{cat}.h5'
		
		if not path.is_file():
			download = A.pull('download', False)
			if download:
				self.download(A)
			else:
				raise MissingDatasetError(f'mpi3d-{cat}')
		
		with hf.File(path, 'r') as f:
			if mode == 'full':
				images = np.concatenate([f['train_images'][()], f['test_images'][()]])
				indices = np.concatenate([f['train_idx'], f['test_idx'][()]])
			elif mode == 'test':
				images = f['test_images'][()]
				indices = f['test_idx'][()]
			else:
				images = f['train_images'][()]
				indices = f['train_idx'][()]
		
		ordered = A.pull('ordered', mode == 'full')
		if ordered:
			self.register_buffer('sel_index', torch.from_numpy(indices.argsort()))
		else:
			self.sel_index = None
		
		self.register_buffer('images', torch.from_numpy(images).permute(0,3,1,2))
		self.register_buffer('indices', torch.from_numpy(indices))

	
	_source_url = {
		'toy': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz',
		'realistic': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz',
		'real': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz',
	}
	@classmethod
	def download(cls, A, dataroot=None, cat=None, **kwargs):
		
		if dataroot is None:
			dataroot = util.get_data_dir(A)
			dataroot = dataroot / 'mpi3d'
		dataroot.mkdir(exist_ok=True)
		
		if cat is None:
			cat = A.pull('category', 'toy')
		
		assert cat in {'toy', 'sim', 'realistic', 'real'}, f'invalid category: {cat}'
		if cat == 'sim':
			cat = 'realistic'
		
		ratio = A.pull('separate-testset', 0.2)
		
		path = dataroot / f'mpi3d_{cat}.h5'
		
		if not path.exists():
			rawpath = dataroot / f'mpi3d_{cat}.npz'
			if not rawpath.exists():
				print(f'Downloading mpi3d-{cat}')
				wget.download(cls._source_url[cat], str(rawpath))
			
			print('Loading full dataset into memory to split train/test')
			full = np.load(rawpath)['images']
			
			N = len(full)
		
			test_N = int(N * ratio)

			rng = np.random.RandomState(0)
			order = rng.permutation(N)
			
			train_idx = order[:-test_N]
			train_idx.sort()
			test_idx = order[-test_N:]
			test_idx.sort()
			
			with hf.File(path, 'w') as f:
				f.create_dataset('train_idx', data=train_idx)
				f.create_dataset('train_images', data=full[train_idx])
				
				f.create_dataset('test_idx', data=test_idx)
				f.create_dataset('test_images', data=full[test_idx])
				
			os.remove(str(rawpath))

	def get_factor_sizes(self):
		return self.factor_sizes
	def get_factor_order(self):
		return self.factor_order

	def get_labels(self):
		return self.get_label(self.indices)

	def update_data(self, indices):
		self.images = self.images[indices]
		if self.labeled:
			self.indices = indices

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
		if self.sel_index is not None:
			idx = self.sel_index[idx]
		imgs = self.images[idx].float().div(255)
		if self.labeled:
			labels = self.get_label(self.indices[idx])
			return imgs, labels
		return imgs,


class OldFullCelebA(Downloadable, ImageDataset):  # TODO: automate downloading and formatting
	
	din = (3, 218, 178)
	
	ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
	              'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
	              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
	              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
	              'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
	              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
	              'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
	              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', ]
	
	def __init__(self, A, resize=unspecified_argument, label_type=unspecified_argument, mode=None, **kwargs):
		self.add_existing_modes('val', 'test')
		
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
		
		if resize is not None:  # TODO: use Interpolated as modifier
			din = 3, *resize
		
		super().__init__(A, din=din, dout=dout, **kwargs)
		self.root = Path(self.root) / 'celeba'
		dataroot = self.root
		name = 'celeba_test.h5' if mode == 'test' else 'celeba_train.h5'
		
		with hf.File(os.path.join(dataroot, 'celeba', name), 'r') as f:
			self.images = f['images'][()]  # encoded as str
			self.labels = f[_labels[label_type]][()] if label_type is not None else None
			
			self.attr_names = f.attrs['attr_names']
			self.landmark_names = f.attrs['landmark_names']
		
		self.resize = resize
	
	# google drive ids
	_google_drive_ids = {
		'img_align_celeba.zip': '0B7EVK8r0v71pZjFTYXZWM3FlRnM',
		'list_eval_partition.txt': '0B7EVK8r0v71pY0NSMzRuSXJEVkk',
		'identity_CelebA.txt': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
		'list_attr_celeba.txt': '0B7EVK8r0v71pblRyaVFSWGxPY0U',
		'list_bbox_celeba.txt': '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
		'list_landmarks_align_celeba.txt': '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
	}
	
	def download(cls, A, dataroot=None, **kwargs):
		
		dataroot = util.get_data_dir(A)
		dataroot = dataroot / 'celeba'
		
		raise NotImplementedError
	
	def get_factor_sizes(self):
		return [2] * len(self.ATTRIBUTES)
	
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
		
		img = torch.from_numpy(util.str_to_jpeg(self.images[item])).permute(2, 0, 1).float().div(255)
		
		if self.resize is not None:
			img = F.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear').squeeze(0)
		
		if self.labels is None:
			return img,
		
		lbl = torch.from_numpy(self.labels[item])
		
		return img, lbl




