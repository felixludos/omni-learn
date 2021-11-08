import sys, os
from pathlib import Path
import subprocess
import pickle
from typing import Any
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
	Downloadable, Dataset, MissingDatasetError, Supervised, Disentanglement, Mechanistic, wrap_dataset

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



@fig.AutoModifier('selected')
class Selected(Splitable):
	def __init__(self, A, ordered=unspecified_argument, unordered=unspecified_argument,
	             reject=unspecified_argument, invert=None, eval_name=None, **kwargs):
		
		if reject is unspecified_argument:
			reject = A.pull('reject', {})
		if ordered is unspecified_argument:
			ordered = A.pull('ordered', None)
		if unordered is unspecified_argument:
			unordered = A.pull('unordered', None)
		
		if invert is None:
			invert = A.pull('invert', False)
		
		if eval_name is None:
			eval_name = A.pull('eval-name', 'extra')
		
		if reject is not None or ordered is not None or unordered is not None:
			A.push('load-labels', True)
		
		self.reject = reject
		self.ordered = ordered
		self.unordered = unordered
		
		self.invert = invert
		
		self.eval_name = eval_name
		
		super().__init__(A, **kwargs)
		
		
	def _split_load(self, dataset):
		
		sizes = self.get_factor_sizes()
		flts = {}
		
		if self.reject is not None:
			flts.update({int(k): v for k, v in self.reject.items()})
		
		if self.ordered is not None:
			for idx, ratio in self.ordered.items():
				idx = int(idx)
				if idx not in flts and ratio != 0:
					N = sizes[idx]
					vals = np.arange(N)
					num = min(np.ceil(abs(ratio) * N), N - 1) if isinstance(ratio, float) \
						else max(1, min(abs(ratio), N - 1))
					num = int(num)
					vals = vals[:num] if ratio > 0 else vals[-num:]
					vals = vals.tolist()
					
					flts[idx] = vals
		
		if self.unordered is not None:
			raise NotImplementedError
		
		self.reject = flts
		
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
		if not self.invert:
			ok = torch.logical_not(ok)
			
		inds = torch.arange(len(lbls), device=lbls.device)
		
		extra = wrap_dataset('subset', dataset, inds[torch.logical_not(ok)])
		self.register_mode(self.eval_name, extra)
		
		dataset = wrap_dataset('subset', dataset, inds[ok])
		self.register_mode(self.split_src, dataset)

		return dataset



@register_dataset('dsprites')
class dSprites(Downloadable, Batchable, ImageDataset, Mechanistic):
	din = (1, 64, 64)
	dout = 5
	# label_space = util.JointSpace()

	def __init__(self, A, slim=None, supervised=None, target_type=unspecified_argument,
	             din=None, dout=None, **kwargs):

		if slim is None:
			slim = A.pull('slim', False)

		if supervised is None:
			supervised = A.pull('supervised', False)

		if target_type is unspecified_argument:
			target_type = A.pull('target_type', 'value' if supervised else None)
		assert target_type in {None, 'value', 'class'}, f'Unknown type of label: {target_type}'

		if din is None: din = A.pull('din', self.din)
		if dout is None: dout = A.pull('dout', self.dout if supervised else din)

		super().__init__(A, supervised=target_type is not None, **kwargs)

		if target_type == 'class':
			self.dout = 113

		dataroot = self.root / 'dsprites'
		dataroot.mkdir(exist_ok=True)
		self.root = dataroot

		path = dataroot / 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
		if not path.is_file():
			download = A.pull('download', False)
			if download:
				self.download(A)
			else:
				raise MissingDatasetError('dsprites')

		print(f'Loading dSprites dataset from disk: {path}')
		data = np.load(path, allow_pickle=True, encoding='bytes')

		self.meta = _rec_decode(data['metadata'][()])

		images = torch.from_numpy(data['imgs']).unsqueeze(1)
		self.register_buffer('images', images)

		if not slim or target_type == 'value':
			self.register_data('value', torch.from_numpy(data['latents_values'][:, 1:]).float())
		if not slim or target_type == 'class':
			self.register_data('class', torch.from_numpy(data['latents_classes'][:, 1:]).int())

		if self.is_supervised():
			self.register_data_aliases('labels', target_type)


	@classmethod
	def download(cls, A, dataroot=None, **kwargs):
		raise NotImplementedError



@register_dataset('3dshapes')
class Shapes3D(Downloadable, Batchable, ImageDataset, Mechanistic):

	din = (3, 64, 64)
	dout = 6

	_full_mechanism_space = util.JointSpace(util.PeriodicDim(), util.PeriodicDim(), util.PeriodicDim(),
		                       util.BoundDim(0.75, 1.25), util.CategoricalDim(4), util.BoundDim(-30., 30.))

	_all_label_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
	_full_label_space = util.JointSpace(util.CategoricalDim(10), util.CategoricalDim(10), util.CategoricalDim(10),
	                                     util.CategoricalDim(8), util.CategoricalDim(4), util.CategoricalDim(15))

	_hue_names = ['red', 'orange', 'yellow', 'green', 'seagreen', 'cyan', 'blue', 'dark-blue', 'purple', 'pink']
	_all_label_class_names = [
		_hue_names, _hue_names, _hue_names,
		list(map(str,range(8))),
		['cube', 'cylinder', 'ball', 'capsule'],
		list(map(str,range(15))),
	]
	del _hue_names

	def __init__(self, A, mode=None, slim=None, supervised=None, din=None, dout=None, **kwargs):

		if slim is None:
			slim = A.pull('slim', False)

		if mode is None:
			mode = A.pull('mode', 'full')
		if supervised is None:
			supervised = A.pull('supervised', False)

		if din is None: din = A.pull('din', self.din)
		if dout is None: dout = A.pull('dout', self.dout if supervised else din)

		super().__init__(A, din=din, dout=dout, supervised=supervised, **kwargs)
		self.add_available_modes('test')
		
		# self.factor_order = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
		# self.factor_sizes = [10, 10, 10, 8, 4, 15]
		# self.factor_num_values = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
		#                           'scale': 8, 'shape': 4, 'orientation': 15}
		
		# raw_mins = torch.tensor([  0.  ,   0.  ,   0.  ,   0.75,   0.  , -30.  ]).float()
		# raw_maxs = torch.tensor([  0.9 ,  0.9 ,  0.9 ,  1.25,  3.  , 30.  ]).float()
		
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

		with hf.File(path, 'r') as data:
			images = torch.from_numpy(data['images'][()])
			labels = torch.from_numpy(data['labels'][()]).float() if not slim or self.is_supervised() else None

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

			images = images.permute(0, 3, 1, 2)
			self.register_buffer('images', images)
			
			if labels is not None:
				self.register_buffer('mechanisms', labels)
				if not slim or not self.uses_mechanisms():
					labels = self.transform_to_labels(labels)
					self.register_buffer('labels', labels)


	def get_images(self, idx=None, **kwargs):
		images = super().get_images(idx=idx, **kwargs)
		return images.float().div(255).clamp(1e-7, 1-1e-7)#.float().div(255)

		
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

	# def get_factor_sizes(self):
	# 	return self.factor_sizes
	# def get_factor_order(self):
	# 	return self.factor_order




@register_dataset('rfd')
class RFD(Downloadable, ImageDataset, Disentanglement):

	""" Implementing RA dataset for RFD based on .h5 storage"""
	shape = (3, 128,128)
	din = shape
	dout = 9
	raw_subfolders = {'train':"finger", 'test':'finger', 'heldout_test':"finger_heldout_colors",
	                  'real_test':"finger_real"}
	raw_files = {'train':8, 'test':2, 'heldout_test':1} #train, test, heldout_test files

	# expected folder structure
	# root (whatever this is it must be passed in the kwargs)
	# |-RFD
	# 	|-raw
	#		|- finger
	#			finger_info.npz
	#		|- finger_heldout
	#			finger_heldout_info.npz
	#		|- finger_real
	#			finger_real_info.npz
	#			finger_real_labels.npz
	#			finger_real_images.npz
	# 	|-processed
	# 		|-processed
	# 		Here go all the .h5 files


	def __init__(self,
				 A,
	             mode=None,
				 labeled=None,
	             din=None, dout=None,
	             **kwargs) -> None:

		if mode is None: mode = A.pull('mode', 'train')
		if labeled is None: labeled = A.pull('labeled', False)

		if din is None: din = A.pull('din', self.din)
		if dout is None: dout = A.pull('dout', self.dout if labeled else din)

		super().__init__(A, din=din, dout=din if dout is None else dout, **kwargs)
		self.root = self.root / 'RFD'
		self.add_available_modes(*self.raw_subfolders.keys())
		
		self.partitions_dict = None
		self.real_set = None
		self.info = None
		
		self.labeled = labeled
		
		self.switch_to(mode)

	def switch_to(self, mode):
		if mode == 'real_test' and self.real_set is None:
			self.real_set = self.read_real_images_and_labels()
		elif mode != self.get_mode() or self.partitions_dict is None:
			self.partitions_dict = self.init_partitions()
		self.info = self.read_info()
		return super().switch_to(mode)

	def read_real_images_and_labels(self):
		""" Loading function only for real test dataset images"""
		print("====== Opening real RFD Dataset ======")
		images = np.load(self.raw_folder+"_images.npz", allow_pickle=True)["images"]
		labels = np.load(self.raw_folder+"_labels.npz", allow_pickle=True)["labels"]
		return (images, labels)

	def init_partitions(self):
		"""Initialising the .h5 files and their order.
		Note: when reading .h5 file it won't be loaded into memory until its
		entries are called."""
		print("====== Opening RFD Dataset ======")
		partitions_dict = {idx: self.open_partition(idx) for idx in range(self.num_files)}
		return partitions_dict

	def get_relative_index(self, index):
		""" Given absolute index it returns the partition index and the
		relative index inside the partition
		#TODO: make it work for multiple indices"""
		len_per_file = self.partitions_dict[0][1]
		partition_idx = index//len_per_file
		relative_index = index%len_per_file
		if index >= len_per_file*self.num_files:
			# contained in last partition
			relative_index = index - len_per_file*self.num_files
		return partition_idx, relative_index

	def open_partition(self, number):
		filename = f"RFD_{self.set_name}_{number}.h5"
		f = hf.File(str(self.root / 'processed' / filename), 'r')
		images = f['images']
		labels = f['labels']
		length = images.shape[0]
		return (images, labels), length

	def __getitem__(self, index: int) -> Any:
		if not isinstance(index, int):
			index = index.item()
		if self.get_mode() == 'real_test':
			imgs, lbls = self.real_set
			rel_idx = index
		else:
			par_idx, rel_idx = self.get_relative_index(index)
			imgs, lbls = self.partitions_dict[par_idx][0]
		img = torch.from_numpy(imgs[rel_idx]).float().div(255).permute(2,0,1)#rescaling the uint8 # from (H,W,C) to (C, H, W)
		
		# both the above are numpy arrays
		# so they need to be cast to torch tensors
		#Note: no rescaling needed as the imgs have already been
		# processed by the ToTensor transformation.
		# if self.transform is not None:
		# 	img = self.transform(img)
		if self.labeled:
			lbl = torch.from_numpy(lbls[rel_idx])
			# if self.target_transform is not None:
			# 	lbl = self.target_transform(lbl)
			return img, lbl
		return img

	def __len__(self) -> int:
		#TODO: see here too for the "batched" case
		if self.get_mode() in {'heldout_test', 'real_test'}: return self.size
		if self.get_mode() != 'test': return int(self.size*0.8)
		# only test is remaining
		return self.size -  int(self.size*0.8)

	def read_info(self):
		""" Opens the labels and info .npz files and stores them in the class"""
		# loading labels
		info = dict(np.load(self.raw_folder+"_info.npz", allow_pickle=True))
		self.size = info["dataset_size"].item()
		self.factor_order = list(info['factor_values'].item().keys())
		self.factor_sizes = info['num_factor_values']
		self.factor_num_values = {k:v for k,v in zip(self.factor_order, self.factor_sizes)}
		return info

	def get_factor_sizes(self):
		return self.factor_sizes

	def get_factor_order(self):
		return self.factor_order

	def get_observation(self):
		raise NotImplementedError

	def get_target(self):
		raise NotImplementedError

	def _update_data(self, indices):
		raise NotImplementedError

	def close_all(self):
		#TODO: use this when finished to close all the hdf5 files opened
		pass

	@property
	def raw_folder(self) -> str:
		prefix = self.raw_subfolders.get(self.get_mode(), 'finger')
		return str(self.root / 'raw' / prefix / prefix)

	@property
	def num_files(self) -> int:
		return self.raw_files.get(self.get_mode(), self.raw_files['train'])

	# @property
	# def processed_folder(self) -> str:
	# 	return self.root + 'RFD/processed/'

	@property
	def set_name(self) -> str:
		if self.get_mode() == 'test': return "test"
		if self.get_mode() == 'heldout_test': return "HC"
		return "train"




@register_dataset('full-celeba')  # probably shouldnt be used
class FullCelebA(Downloadable, ImageDataset, Disentanglement):  # TODO: automate downloading and formatting
	
	din = (3, 218, 178)

	_all_label_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
	              'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
	              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
	              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
	              'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
	              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
	              'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
	              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', ]
	_all_label_class_names = [[f'no:{l}', f'yes:{l}'] for l in _all_label_names]
	_full_label_space = util.JointSpace(*[util.BinaryDim() for _ in range(len(_all_label_names))])


	def __init__(self, A, resize=unspecified_argument, slim=None, supervised=None,
	             target_type=unspecified_argument, mode=None,
	             din=None, dout=None, **kwargs):

		if slim is None:
			slim = A.pull('slim', False)

		if mode is None:
			mode = A.pull('mode', 'train')

		if supervised is None:
			supervised = A.pull('supervised', False)
		if target_type is unspecified_argument:
			target_type = A.pull('target_type', 'attr' if supervised else None)

		if resize is unspecified_argument:
			resize = A.pull('resize', (256, 256))
		if resize is not None:  # TODO: use Interpolated as modifier
			din = 3, *resize

		if din is None: din = A.pull('din', self.din)
		if dout is None: dout = A.pull('dout', self.dout if supervised else din)

		if dout is None:
			if target_type is None:
				dout = din
			elif target_type == 'attr':
				dout = 40
			elif target_type == 'landmark':
				dout = 10
			elif target_type == 'identity':
				dout = 1
			else:
				raise Exception(f'unknown {target_type}')

			dout = A.pull('dout', dout)

		_labels = {
			'attr': 'attrs',
			'identity': 'identities',
			'landmark': 'landmarks',
		}

		super().__init__(A, din=din, dout=dout, supervised=supervised, **kwargs)
		self.add_available_modes('test')

		dataroot = self.root / 'celeba'
		dataroot.mkdir(exist_ok=True)
		self.root = dataroot
		filename = 'celeba_test.h5' if mode == 'test' else 'celeba_train.h5'

		with hf.File(dataroot/filename, 'r') as f:
			self.images = f['images'][()]  # encoded as str

			for key, fkey in _labels.items():
				if not slim or target_type == key:
					self.register_data(key, torch.from_numpy(f[fkey][()]))

			self.attr_names = f.attrs['attr_names']
			self.landmark_names = f.attrs['landmark_names']

		if self.is_supervised():
			self.register_data_aliases(target_type, 'labels')

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


	def _load_jpeg_image(self, idx):
		img = torch.from_numpy(util.str_to_jpeg(self.images[idx])).permute(2, 0, 1).float().div(255).clamp(1e-7, 1-1e-7)
		if self.resize is not None:
			img = F.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear').squeeze(0)
		return img


	def get_images(self, idx=None):
		if idx is None:
			return torch.stack([self._load_jpeg_image(i) for i in range(len(self))])
		return self._load_jpeg_image(idx)



@register_dataset('celeba')
class CelebA(Cropped, FullCelebA):
	def __init__(self, A, resize=None, crop_size=128, **kwargs):
		# raise NotImplementedError('doesnt work since FullCelebA automatically resizes to (256,256)')
		crop_size = A.pull('crop_size', crop_size) # essentially sets this as the default

		super().__init__(A, resize=resize, crop_size=crop_size, **kwargs)



@register_dataset('mpi3d')
class MPI3D(Downloadable, Batchable, ImageDataset, Mechanistic):

	din = (3, 64, 64)
	dout = 7

	_all_label_names = ['object_color', 'object_shape', 'object_size', 'camera_height', 'background_color',
		                     'horizonal_axis', 'vertical_axis']
	_all_label_class_names = [
		['white', 'green', 'red', 'blue', 'brown', 'olive'],
		['cone', 'cube', 'cylinder', 'hexagonal', 'pyramid', 'sphere'],
		['small', 'large'],
		['top', 'center', 'bottom'],
		['purple', 'sea_green', 'salmon'],
		list(map(str,range(40))), list(map(str,range(40))),
	]
	_full_mechanism_space = util.JointSpace(util.CategoricalDim(6), util.CategoricalDim(6), util.BoundDim(),
		                       util.BoundDim(0,1), util.CategoricalDim(3),
		                       util.BoundDim(0,1), util.BoundDim(0,1))
	_full_label_space = util.JointSpace(util.CategoricalDim(6), util.CategoricalDim(6), util.CategoricalDim(2),
	                                util.CategoricalDim(3), util.CategoricalDim(3),
	                                util.CategoricalDim(40), util.CategoricalDim(40))


	def __init__(self, A, slim=None, mode=None, fid_ident=None,
	             supervised=None, din=None, dout=None, **kwargs):

		if slim is None:
			slim = A.pull('slim', False)

		if mode is None:
			mode = A.pull('mode', 'train')
		if supervised is None:
			supervised = A.pull('supervised', False)

		if din is None: din = A.pull('din', self.din)
		if dout is None: dout = A.pull('dout', self.dout if supervised else din)

		cat = A.pull('category', 'toy')
		assert cat in {'toy', 'sim', 'realistic', 'real', 'complex'}, f'invalid category: {cat}'
		if cat == 'sim':
			cat = 'realistic'

		super().__init__(A, din=din, dout=dout, supervised=supervised, fid_ident=cat, **kwargs)
		self.add_available_modes('test')
		dataroot = self.root / 'mpi3d'
		self.root = dataroot

		if cat == 'complex':
			self._all_label_class_names[0] = ['mug', 'ball', 'banana', 'cup']
			self._all_label_class_names[1] = ['yellow', 'green', 'olive', 'red']

			self._full_mechanism_space = util.JointSpace(util.CategoricalDim(4), util.CategoricalDim(4),
			                                             util.BoundDim(), util.BoundDim(), util.CategoricalDim(3),
			                                        util.BoundDim(), util.BoundDim())
			self._full_label_space = util.JointSpace(util.CategoricalDim(4), util.CategoricalDim(4),
			                                         util.CategoricalDim(2), util.CategoricalDim(3),
			                                         util.CategoricalDim(3), util.CategoricalDim(40),
			                                         util.CategoricalDim(40))

		sizes = np.array(self.get_label_sizes())
		flr = np.cumprod(sizes[::-1])[::-1]
		flr[:-1] = flr[1:]
		flr[-1] = 1
		self._sizes = torch.from_numpy(sizes.copy()).long()
		self._flr = torch.from_numpy(flr.copy()).long()

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
				indices = np.concatenate([f['train_idx'][()], f['test_idx'][()]])
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


	def _get_label(self, inds):
		try:
			len(inds)
			inds = inds.view(-1,1)
		except TypeError:
			pass

		lvls = inds // self._flr
		labels = lvls % self._sizes
		return labels


	def get_labels(self, idx=None):
		if idx is None:
			inds = self.indices
			if self.sel_index is not None:
				inds = inds[self.sel_index]
		else:
			if self.sel_index is not None:
				idx = self.sel_index[idx]
			inds = self.indices[idx]
		return self._get_label(inds)


	def get_images(self, idx=None):
		if idx is None:
			imgs = self.images
			if self.sel_index is not None:
				imgs = imgs[self.sel_index]
		else:
			if self.sel_index is not None:
				idx = self.sel_index[idx]
			imgs = self.images[idx]
		return imgs.float().div(255).clamp(1e-7, 1 - 1e-7)


	# def __getitem__(self, idx):
	# 	if self.sel_index is not None:
	# 		idx = self.sel_index[idx]
	# 	imgs = self.images[idx].float().div(255).clamp(1e-7, 1-1e-7)
	# 	if self.labeled:
	# 		labels = self.get_label(self.indices[idx])
	# 		return imgs, labels
	# 	return imgs,








#
# class OldFullCelebA(Downloadable, ImageDataset):  # TODO: automate downloading and formatting
#
# 	din = (3, 218, 178)
#
# 	ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
# 	              'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
# 	              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
# 	              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
# 	              'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
# 	              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
# 	              'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
# 	              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', ]
#
# 	def __init__(self, A, resize=unspecified_argument, label_type=unspecified_argument, mode=None, **kwargs):
# 		self.add_existing_modes('val', 'test')
#
# 		if label_type is unspecified_argument:
# 			label_type = A.pull('label_type', None)
#
# 		if mode is None:
# 			mode = A.pull('mode', 'train')
# 		if resize is unspecified_argument:
# 			resize = A.pull('resize', (256, 256))
#
# 		din = A.pull('din', self.din)
#
# 		if label_type is None:
# 			dout = din
# 		elif label_type == 'attr':
# 			dout = 40
# 		elif label_type == 'landmark':
# 			dout = 10
# 		elif label_type == 'identity':
# 			dout = 1
# 		else:
# 			raise Exception('unknown {}'.format(label_type))
#
# 		dout = A.pull('dout', dout)
#
# 		_labels = {
# 			'attr': 'attrs',
# 			'identity': 'identities',
# 			'landmark': 'landmarks',
# 		}
#
# 		if resize is not None:  # TODO: use Interpolated as modifier
# 			din = 3, *resize
#
# 		super().__init__(A, din=din, dout=dout, **kwargs)
# 		self.root = Path(self.root) / 'celeba'
# 		dataroot = self.root
# 		name = 'celeba_test.h5' if mode == 'test' else 'celeba_train.h5'
#
# 		with hf.File(os.path.join(dataroot, 'celeba', name), 'r') as f:
# 			self.images = f['images'][()]  # encoded as str
# 			self.labels = f[_labels[label_type]][()] if label_type is not None else None
#
# 			self.attr_names = f.attrs['attr_names']
# 			self.landmark_names = f.attrs['landmark_names']
#
# 		self.resize = resize
#
# 	# google drive ids
# 	_google_drive_ids = {
# 		'img_align_celeba.zip': '0B7EVK8r0v71pZjFTYXZWM3FlRnM',
# 		'list_eval_partition.txt': '0B7EVK8r0v71pY0NSMzRuSXJEVkk',
# 		'identity_CelebA.txt': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
# 		'list_attr_celeba.txt': '0B7EVK8r0v71pblRyaVFSWGxPY0U',
# 		'list_bbox_celeba.txt': '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
# 		'list_landmarks_align_celeba.txt': '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
# 	}
#
# 	def download(cls, A, dataroot=None, **kwargs):
#
# 		dataroot = util.get_data_dir(A)
# 		dataroot = dataroot / 'celeba'
#
# 		raise NotImplementedError
#
# 	def get_factor_sizes(self):
# 		return [2] * len(self.ATTRIBUTES)
#
# 	def get_factor_order(self):
# 		return self.ATTRIBUTES
#
# 	def get_attribute_key(self, idx):
# 		try:
# 			return self.ATTRIBUTES[idx]
# 		except IndexError:
# 			pass
#
# 	def __len__(self):
# 		return len(self.images)
#
# 	def __getitem__(self, item):
#
# 		img = torch.from_numpy(util.str_to_jpeg(self.images[item])).permute(2, 0, 1).float().div(255).clamp(1e-7, 1-1e-7)
#
# 		if self.resize is not None:
# 			img = F.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear').squeeze(0)
#
# 		if self.labels is None:
# 			return img,
#
# 		lbl = torch.from_numpy(self.labels[item])
#
# 		return img, lbl
#
#


