
import sys, os
from pathlib import Path

import random

from tqdm import tqdm

from omnibelt import create_dir, get_printer

prt = get_printer(__name__)

try:
	import wget
except ImportError:
	prt.warning('wget not found')
import shutil
import h5py as hf

import cv2
import numpy as np
import torch

import omnifig as fig

from ... import util

from ...data import register_dataset, Deviced, Batchable, ImageDataset

DATASET_URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'

def get_images(root, size=None, is_mask=False):
	raw = [util.read_raw_bytes(name, root=root) for name in os.listdir(root)]
	if size is None:
		return raw
	
	images = []
	for data in raw:
		image = util.str_to_byte_img(data) if is_mask else util.str_to_rgb(data)[..., ::-1]
		image = cv2.resize(image, (size,size))
		images.append(image)
	return images
	
@fig.Script('download-mvtec', description='Download and Format MVTec Anomaly Detection dataset')
def download_mvtec(A):
	raise NotImplementedError
	root = A.pull('root', os.environ.get('OMNILEARN_DATA_DIR', None))
	
	if root is None:
		raise Exception('Must specify a OMNILEARN_DATA_DIR as env variable (or pass in a `root`)')
	
	root = os.path.join(root, 'mvtec')
	
	if not os.path.isdir(root):
		create_dir(root)

	raw_name = A.pull('raw-name', 'mvtec_anomaly_detection.tar.xz', silent=True)
	
	size = A.pull('size', None)
	name = 'full' if size is None else f'size{size}'
	
	path = os.path.join(root, name)
	
	cleanup = A.pull('cleanup', False)
	cleanup_compressed = A.pull('cleanup-compressed', False)
	
	base = os.path.join(root, 'raw')
	if not os.path.isdir(base):
		rawpath = os.path.join(root, raw_name)
		if not os.path.isfile(rawpath):
			url = A.pull('url', DATASET_URL)
			print(f'Downloading MVTec AD dataset from {url}')
			
			rawpath = wget.download(url, out=root)
		
		create_dir(base)
		
		print(f'Extracting MVTec AD dataset from {rawpath} file')
		
		shutil.unpack_archive(rawpath, extract_dir=base)
		
		if cleanup_compressed:
			os.remove(rawpath)
	
	create_dir(path)
	
	contents = os.listdir(path)
	
	todo = [name for name in os.listdir(base)
	        if os.path.isdir(os.path.join(base, name))
	        and f'{name}.h5' not in contents]
	
	print(f'Formatting {len(todo)} object directories')
	
	pbar = A.pull('pbar', True, silent=True)
	if pbar:
		todo = tqdm(todo)
	
	for cat in todo:
		
		croot = os.path.join(base, cat)
		
		if pbar:
			todo.set_description(cat)
		
		with hf.File(os.path.join(path, f'{cat}.h5'), 'w') as f:
		
			mroot = os.path.join(croot, 'ground_truth')
		
			defects = os.listdir(mroot)
			f.attrs['defects'] = defects
		
			for defect in defects:
				f.create_dataset(f'mask_{defect}', data=get_images(os.path.join(mroot, defect),
				                                                   size=size, is_mask=True))
		
			f.create_dataset('train_good', data=get_images(os.path.join(croot, 'train', 'good'), size=size))
			
			mroot = os.path.join(croot, 'test')
			for test in os.listdir(mroot):
				f.create_dataset(f'test_{test}', data=get_images(os.path.join(mroot, test), size=size))
		
		if cleanup:
			print(f'removing: {cat}')
			# shutil.rmtree(croot)
	
	print('Formatted all categories')
	
	return path

@register_dataset('mvtec')
class MVTec_Anomaly_Detection(Batchable):

	CATEGORIES = {'bottle', 'carpet', 'leather', 'screw', 'transistor', 'cable', 'grid',
        'metal_nut', 'tile', 'wood', 'capsule', 'hazelnut', 'pill','toothbrush', 'zipper'}
	GREYSCALES = {'screw', 'grid', 'zipper'}

	def __init__(self, A, **kwargs):
	
		dataroot = util.get_data_dir(A) / 'mvtec'

		mode = A.pull('mode', 'train')

		size = A.pull('size', None)
		
		dirname = 'full' if size is None else f'size{size}'
		droot = dataroot / dirname
		
		if not dataroot.is_dir() or not droot.is_dir():
			download = A.pull('download', False)
			if not download:
				raise Exception(f'This dataset ({dirname}) hasnt been downloaded and setup yet '
				                '(set the "download" argument to do so automatically)')
		
			print(f'Downloading/Formatting MVTec dataset: {dirname}')
			droot = fig.quick_run('download-mvtec', root=str(dataroot), size=size)
			droot = Path(droot)
		
		cat = A.pull('cat', 'random')
		if cat == 'random':
			cat = random.choice(list(self.CATEGORIES))

		ratio = A.pull('ratio', None)
		if ratio is not None:
			assert 0 < ratio <= 1, f'bad ratio: {ratio}'

		path = [c for c in droot.glob('*.h5') if cat == c.stem][0]
		
		include_class = A.pull('include-class', '<>include_class', True)
		include_mask = A.pull('include-masks', '<>include_masks', False)
		
		cut = A.pull('cut', 'train')
		assert cut in {None, 'normal', 'anomalies', 'all', 'train', 'test'}, f'unknown: {cut}'

		tfms = A.pull('transforms', None, silent=True)
		if tfms is not None:
			tfms = tfms.get(cat, None)

		augmenter = None
		augment_factor = 1
		if tfms is not None:
			augment_here = A.pull('augment_factor', None)

			if augment_here is not None:
				A.push('augmenter', tfms)
				A.push('augmenter._type', 'image-transform', overwrite=False)
				augmenter = A.pull('augmenter', None)
				augment_factor = augment_here
			else:
				A.push('augmentations', tfms, force_root=True)



		C = 1 if cat in self.GREYSCALES else 3
		din = (C, 1024, 1024) if size is None else (C, size, size)
		dout = din
		super().__init__(A, din=din, dout=dout, **kwargs)
		
		raw = hf.File(path, 'r')
		
		# print(raw.keys())

		uses = [key for key in raw.keys() if 'mask_' not in key]
		if cut == 'train':
			uses = [key for key in uses if 'train_' in key]
		elif cut == 'test':
			uses = [key for key in uses if 'test_' in key]
		elif cut == 'normal':
			uses = [key for key in uses if 'train_' in key or key == 'test_good']
		elif cut == 'anomalies':
			uses = [key for key in uses if 'test_' in key and key != 'test_good']

		C, H, W = din
		
		images = []
		masks = [] if include_mask else None
		labels = [] if include_class else None
		idents = []
		for key in uses:
			N = len(raw[key])
			if ratio is None:
				sel = ()
			elif mode == 'test':
				sel = slice(int(-N*(1-ratio)),None)
			else:
				sel = slice(int(N*ratio))

			imgs = raw[key][sel]
			# print(key, imgs.shape)
			images.extend(imgs)
			if labels is not None:
				labels.extend([1 if 'test_' in key and key != 'test_good' else 0]*len(imgs))
			ident = '_'.join(key.split('_')[1:])
			idents.append(ident)
			if masks is not None:
				if 'test_' in key and ident != 'good':
					ident = f'mask_{ident}'
					masks.extend(raw[ident][sel])
				else:
					masks.extend([np.zeros((H,W))]*len(imgs))
		
		if masks is not None and not len(masks):
			masks = None
		if labels is not None:
			labels = torch.tensor(labels).long()
		
		if size is not None:
			# print([m.shape for m in images])
			images = torch.from_numpy(np.stack(images))
			images = images.unsqueeze(1) if C == 1 else images.permute(0,3,1,2)
			images = images.float().div(255)
			if masks is not None and len(masks):
				# print([m.shape for m in masks])
				masks = torch.from_numpy(np.stack(masks)).unsqueeze(1).bool()

		raw.close()
		# self.raw = raw

		self.transforms = tfms
		
		self.size = size
		
		self.images = images
		self.masks = masks
		self.labels = labels
		self.uses = uses
		self.idents = idents

		self.augmenter = augmenter
		self.augment_factor = augment_factor

	def get_transforms(self):
		return self.transforms
	
	def __len__(self):
		return len(self.images)#*self.augment_factor
	
	def __getitem__(self, item):

		# item %= len(self.images)
		
		if self.size is not None:
			out = [self.images[item]]

			if self.masks is not None:
				out.append(self.masks[item])

				if self.augmenter is not None:
					full = self.augmenter(torch.cat(out, 1))
					out = [full[:, :-1].contiguous(), full[:, -1:].contiguous()]

				if self.labels is not None:
					out.append(self.labels[item])
			elif self.labels is not None:
				out.append(self.labels[item])
			if self.augmenter is not None:
				out[0] = self.augmenter(out[0])

			return out
		
		raise NotImplementedError




