import os
from omnibelt import agnostic, agnosticproperty
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import subprocess
import wget
import h5py as hf

from omniplex import hparam, material, inherit_hparams
from omniplex import util, spaces, flavors

from ..novo.base import DataProduct, DataBuilder


class DownloadableHDFImages(flavors.DownloadableRouter):
	_image_key_name = 'images'

	ImageBuffer = flavors.ImageBuffer

	@agnostic
	def is_downloaded(self):
		return self.get_archive_path().is_file()


	@agnostic
	def get_archive_path(self):
		return self.root / f'{self.name}.h5'


	@staticmethod
	def _download_source_hdf(dest):
		raise NotImplementedError


	@agnostic
	def download(self, *, testset_ratio=0.2, test_seed=0, force_download=False):
		if self.is_downloaded() and not force_download:
			print('Already downloaded (use force_download=True to overwrite)')
		dest = self.get_archive_path()
		self._download_source_hdf(dest)

		rng = np.random.RandomState(test_seed)
		assert testset_ratio is not None and testset_ratio > 0, 'bad testset ratio'

		with hf.File(dest, 'r+') as f:
			N = f[self._image_key_name].shape[0]

			test_N = int(N * testset_ratio)

			order = rng.permutation(N)

			train_idx = order[:-test_N]
			train_idx.sort()
			test_idx = order[-test_N:]
			test_idx.sort()

			f.create_dataset('train_idx', data=train_idx)
			f.create_dataset('test_idx', data=test_idx)



class DisentanglementData(DataBuilder, ident='disentanglement', as_branch=True):
	pass



class MPI3D(DisentanglementData, ident='mpi3d', as_branch=True):
	pass



class _Synthetic_Disentanglement(flavors.SyntheticDataset, DownloadableHDFImages, DataBuilder,
                                 registry=DisentanglementData):
	@material('mechanism')
	def get_mechanism(self, src):
		return self.transform_to_mechanisms(src['label'])



class dSprites(_Synthetic_Disentanglement, ident='dsprites'):
	_dirname = 'dsprites'
	def __init__(self, default_len=None, as_bytes=False, **kwargs):
		# if default_len is None:
		# 	default_len = 737280
		super().__init__(default_len=default_len, **kwargs)

		self.register_buffer('observation', self.ImageBuffer(space=spaces.Pixels(1, 64, 64, as_bytes=as_bytes)))

		_shape_names = ['square', 'ellipse', 'heart']
		_dim_names = ['shape', 'scale', 'orientation', 'posX', 'posY']
		self.register_buffer('label', space=spaces.Joint(spaces.Categorical(_shape_names),
		                                                 spaces.Categorical(6),
		                                                 spaces.Categorical(40),
		                                                 spaces.Categorical(32),
		                                                 spaces.Categorical(32),
		                                                 names=_dim_names))
		self.get_buffer('mechanism').space = spaces.Joint(spaces.Categorical(_shape_names),
		                                                  spaces.Bound(0.5, 1.),
		                                                  spaces.Periodic(period=2 * np.pi),
		                                                  spaces.Bound(0., 1.),
		                                                  spaces.Bound(0., 1.),
		                                                  names=_dim_names)


	_source_url = 'https://github.com/deepmind/dsprites-dataset/raw/master/' \
	              'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'
	_image_key_name = 'imgs'


	@classmethod
	def _download_source_hdf(cls, dest):
		wget.download(cls._source_url, str(dest))


	# @classmethod
	# def _decode_meta_info(cls, obj):
	# 	'''
	# 	recursively convert bytes to str
	# 	:param obj: root obj
	# 	:return:
	# 	'''
	# 	if isinstance(obj, dict):
	# 		return {cls._decode_meta_info(k): cls._decode_meta_info(v) for k, v in obj.items()}
	# 	if isinstance(obj, list):
	# 		return [cls._decode_meta_info(x) for x in obj]
	# 	if isinstance(obj, tuple):
	# 		return tuple(cls._decode_meta_info(x) for x in obj)
	# 	if isinstance(obj, bytes):
	# 		return obj.decode()
	# 	return obj


	def _prepare(self, *args, **kwargs):
		super()._prepare(*args, **kwargs)

		dest = self.get_archive_path()

		# data = np.load(path, allow_pickle=True, encoding='bytes')
		# self.meta = self._decode_meta_info(data['metadata'][()])

		with hf.File(str(dest), 'r') as f:
			self.get_buffer('observation').data = torch.from_numpy(f[self._image_key_name][()]).unsqueeze(1)
			# self.get_buffer('label').data = torch.from_numpy(data['latents_values'][:, 1:]).float()
			self.get_buffer('label').data = torch.from_numpy(f['latents_classes'][:, 1:]).float()



class Shapes3D(_Synthetic_Disentanglement, ident='shapes3d'):
	_dirname = '3dshapes'
	_default_lens = {'train': 384000, 'test': 96000, 'full': 480000}

	def __init__(self, default_len=None, as_bytes=False, mode=None, **kwargs):
		if default_len is None:
			if mode is None:
				mode = 'train'
			default_len = self._default_lens[mode]
		super().__init__(default_len=default_len, **kwargs)

		self.mode = mode

		_all_label_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
		_hue_names = ['red', 'orange', 'yellow', 'green', 'seagreen', 'cyan', 'blue', 'dark-blue', 'purple', 'pink']
		_shape_names = ['cube', 'cylinder', 'ball', 'capsule']

		self.register_buffer('observation', self.ImageBuffer(space=spaces.Pixels(3, 64, 64, as_bytes=as_bytes)))
		self.register_buffer('label', space=spaces.Joint(spaces.Categorical(_hue_names),
		                                                 spaces.Categorical(_hue_names),
		                                                 spaces.Categorical(_hue_names),
		                                                 spaces.Categorical(8),
		                                                 spaces.Categorical(_shape_names),
		                                                 spaces.Categorical(15),
		                                                 names=_all_label_names))
		self.get_buffer('mechanism').space = spaces.Joint(spaces.Periodic(),
		                                                  spaces.Periodic(),
		                                                  spaces.Periodic(),
		                                                  spaces.Bound(0.75, 1.25),
		                                                  spaces.Categorical(_shape_names),
		                                                  spaces.Bound(-30., 30.),
		                                                  names=_all_label_names)

	_source_url = 'gs://3d-shapes/3dshapes.h5'
	_image_key_name = 'images'

	@agnostic
	def download(self, uncompress=False, **kwargs):
		super().download(**kwargs)

		if uncompress: # TODO: add logging
			arch = self.get_archive_path()
			new = arch.parents[0] / 'uncompressed.h5'

			with hf.File(str(new), 'w') as f:
				with hf.File(str(arch), 'r') as old:
					for key in old.keys():
						f.create_dataset(key, data=old[key][()])
					for key in old.attrs:
						f.attrs[key] = old.attrs[key]

			os.remove(str(arch))
			os.rename(str(new), str(arch))


	@classmethod
	def _download_source_hdf(cls, dest):
		# print(f'Downloading 3dshapes dataset to {str(dest)} ...', end='')
		subprocess.run(['gsutil', 'cp', cls._source_url, str(dest)])
		# print(' done!')


	def _prepare(self, *args, **kwargs):
		super()._prepare(*args, **kwargs)

		mode = self.mode

		dest = self.get_archive_path()
		with hf.File(str(dest), 'r') as f:
			indices = None if mode == 'full' else \
				(f['test_idx'][()] if mode == 'test' else f['train_idx'][()])

			images = f[self._image_key_name][()]
			mechanism = f['labels'][()]
			if indices is not None:
				images = images[indices]
				mechanism = mechanism[indices]

			# images = f[self._image_key_name][indices] # TODO: why is h5py so slow with indexed reads
			images = images.transpose(0, 3, 1, 2)

			self.get_buffer('observation').data = torch.from_numpy(images)
			self.get_buffer('mechanism').data = torch.from_numpy(mechanism).float()



class _MPI3D_Base(_Synthetic_Disentanglement, registry=MPI3D): # TODO
	_dirname = 'mpi3d'

	cat = hparam(space=spaces.Categorical(['toy', 'sim', 'real']))

	def __init__(self, default_len=None, as_bytes=False, **kwargs):
		# if default_len is None:
		# 	default_len = 480000
		super().__init__(default_len=default_len, **kwargs)
		cat = self.cat
		assert cat in {'toy', 'sim', 'real', 'complex'}, f'invalid category: {cat}'
		# self.category = cat

		_all_label_names = ['object_color', 'object_shape', 'object_size', 'camera_height', 'background_color',
		                    'horizonal_axis', 'vertical_axis']
		_colors = ['white', 'green', 'red', 'blue', 'brown', 'olive']
		_shapes = ['cone', 'cube', 'cylinder', 'hexagonal', 'pyramid', 'sphere']
		_bg_color = ['purple', 'sea_green', 'salmon']
		if cat == 'complex':
			_shapes = ['mug', 'ball', 'banana', 'cup']
			_colors = ['yellow', 'green', 'olive', 'red']

		self.register_buffer('observation', self.ImageBuffer(space=spaces.Pixels(3, 64, 64, as_bytes=as_bytes)))
		self.register_buffer('label', space=spaces.Joint(spaces.Categorical(_colors),
		                                                 spaces.Categorical(_shapes),
		                                                 spaces.Categorical(['small', 'large']),
		                                                 spaces.Categorical(['top', 'center', 'bottom']),
		                                                 spaces.Categorical(_bg_color),
		                                                 spaces.Categorical(40),
		                                                 spaces.Categorical(40),
		                                                 names=_all_label_names))
		self.get_buffer('mechanism').space = spaces.Joint(spaces.Categorical(_colors),
		                                                  spaces.Categorical(_shapes),
		                                                  spaces.Bound(0., 1.),
		                                                  spaces.Bound(0., 1.),
		                                                  spaces.Categorical(_bg_color),
		                                                  spaces.Bound(0., 1.),
		                                                  spaces.Bound(0., 1.),
		                                                  names=_all_label_names)

	_source_url = {
		'toy': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz',
		'sim': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz',
		'real': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz',
	}

	@agnosticproperty
	def root(self):
		return self._infer_root() / 'mpi3d'

	@classmethod
	def download(cls, category=None, **kwargs):

		raise NotImplementedError

		if dataroot is None:
			dataroot = util.get_data_dir(A)
			dataroot = dataroot / 'mpi3d'
		dataroot.mkdir(exist_ok=True)

		if cat is None:
			cat = A.pull('category', 'toy')

		assert cat in {'toy', 'sim', 'real'}, f'invalid category: {cat}'

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



class MPI3D_Toy(_MPI3D_Base, ident='toy'):
	cat = 'toy'

class MPI3D_Sim(_MPI3D_Base, ident='sim'):
	cat = 'sim'



class CelebA(flavors.SupervisedDataset, DownloadableHDFImages):
	_dirname = 'celeba'

	def __init__(self, target_type='attr',
	             crop_size=128, resize=None, resize_mode='bilinear', # these args shouldn't be changed
	             as_bytes=False, mode=None, **kwargs):

		raise NotImplementedError # not ready

		super().__init__(mode=mode, **kwargs)
		assert target_type in {'attr', 'identity', 'landmark'}, f'unknown: {target_type}'
		if target_type != 'attr':
			raise NotImplementedError
		self.target_type = target_type

		_labels_dataset_keys = {
			'attr': 'attrs',
			'identity': 'identities',
			'landmark': 'landmarks',
		}

		img_space = spaces.Pixels(3, 218, 178, as_bytes=as_bytes)
		if crop_size is not None:
			if isinstance(crop_size, int):
				crop_size = crop_size, crop_size
			img_space.height, img_space.width = crop_size
		if resize is not None:
			if isinstance(resize, int):
				resize = resize, resize
			img_space.height, img_space.width = resize

		_all_attr_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
	              'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
	              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
	              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
	              'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
	              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
	              'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
	              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', ]
		target_space = spaces.Joint(*[spaces.Binary() for _ in _all_attr_names], names=_all_attr_names)

		root = self.get_root()
		filename = 'celeba_test.h5' if mode == 'test' else 'celeba_train.h5'
		path = root / filename

		img_buffer = self.register_buffer('observation',
		                                  self.ImageBuffer(dataset_name='images', path=path, crop=crop_size,
		                                                     resize=resize, resize_mode=resize_mode),
		                                  space=img_space)
		self.register_buffer('target', HDFBuffer(dataset_name=_labels_dataset_keys[self.target_type], path=path,
		                                         dtype='float'),
		                     space=target_space)

		self._default_len = len(img_buffer)


	class ImageBuffer(DownloadableHDFImages.ImageBuffer):
		def __init__(self, crop=None, crop_base=None,
		             resize=None, resize_mode='bilinear', epsilon=1e-8, **kwargs):
			super().__init__(**kwargs)
			self.crop = crop
			# if crop is not None and crop[0] != crop[1]:
			# 	raise NotImplementedError

			self.resize = resize
			self.resize_mode = resize_mode
			self._epsilon = epsilon


		def _prepare_sel(self, sel=None, **kwargs):
			if sel is None:
				sel = ()
			else:
				if self._selected is not None:
					sel = self._selected[sel]
				sel = torch.as_tensor(sel).numpy()

			with hf.File(str(self.path), 'r') as f:
				sample = f[self.key_name][sel]
			self.data = sample # list of jpeg strings


		def _load_jpeg_image(self, idx):
			img = torch.from_numpy(util.str_to_jpeg(self.data[idx])).permute(2, 0, 1)
			return img


		def _get(self, sel=None, **kwargs):
			if sel is None:
				sel = torch.arange(len(self))
			images = torch.stack([self._load_jpeg_image(i) for i in sel.tolist()])
			if self.crop is not None:
				cx, cy = images.shape[-2]//2, images.shape[-1]//2
				rx, ry = self.crop[0]//2, self.crop[1]//2
				images = images[..., cx-rx:cx+rx, cy-ry:cy+ry]
			if self.resize is not None:
				images = F.interpolate(images.float(), size=self.resize, mode=self.resize_mode)
			# images = images.continguous()
			images = images.byte() if self.space.as_bytes \
				else images.float().div(255).clamp(self._epsilon, 1-self._epsilon)
			return images

	#  TODO: setup download celeba

	# # google drive ids
	# _google_drive_ids = {
	# 	'list_eval_partition.txt': '0B7EVK8r0v71pY0NSMzRuSXJEVkk',
	# 	'identity_CelebA.txt': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
	# 	'list_attr_celeba.txt': '0B7EVK8r0v71pblRyaVFSWGxPY0U',
	# 	'list_bbox_celeba.txt': '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
	# 	'list_landmarks_align_celeba.txt': '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
	# }
	# _google_drive_image_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
	#
	# @classmethod
	# def download(cls, A, dataroot=None, **kwargs):
	#
	# 	raise NotImplementedError('Unfortunately, automatically downloading CelebA is current not supported.')
	#
	# 	dataroot = util.get_data_dir(A)
	# 	dataroot = dataroot / 'celeba'
	# 	dataroot.mkdir(exist_ok=True)
	#
	# 	prt.warning(
	# 		'Downloading CelebA doesn\'t seem to work currently (issues with google drive), instead it must be downloaded manually:'
	# 		'\n1. create a directory "celeba" in the local_data directory'
	# 		'\n2. download "img_align_celeba.zip" into that directory'
	# 		'\n3. download label files: {}'
	# 		'\n4. extract "img_align_celeba.zip" to a directory "img_align_celeba"'.format(
	# 			', '.join(f'"{c}"' for c in cls._google_drive_ids.keys())))
	#
	# 	raise NotImplementedError
	#
	# 	imgdir = dataroot / 'img_align_celeba'
	#
	# 	pbar = A.pull('pbar', None)
	#
	# 	util.download_file_from_google_drive('0B7EVK8r0v71pOXBhSUdJWU1MYUk', dataroot, pbar=pbar)
	#
	# 	for name, ID in cls._google_drive_ids.items():
	# 		path = dataroot / name
	# 		if not path.exists():
	# 			util.download_file_from_google_drive(ID, dataroot, name, pbar=pbar)
	#
	# 	if not imgdir.is_dir():
	#
	# 		imgpath = dataroot / 'img_align_celeba.zip'
	#
	# 		# download zip
	# 		if not imgpath.exists():
	# 			imgpath = util.download_file_from_google_drive(cls._google_drive_image_id, dataroot, pbar=pbar)
	#
	# 		# extract zip
	# 		imgdir.mkdir(exist_ok=True)
	# 		with zipfile.ZipFile(str(imgpath), 'r') as zip_ref:
	# 			zip_ref.extractall(str(imgdir))
	#
	# 	# os.remove(str(imgpath))
	#
	# 	raise NotImplementedError













