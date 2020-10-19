
import sys, os

from tqdm import tqdm
try:
	import wget
except ImportError:
	print('WARNING: unable to import wget')
import shutil
import h5py as hf

import cv2
import numpy as np
import torch

from omnibelt import create_dir
import omnifig as fig

from ... import util
from ..data import Dataset

from ...data import standard_split, Device_Dataset, Info_Dataset, Splitable_Dataset, Testable_Dataset, Batchable_Dataset, Image_Dataset

DATASET_URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'

def get_images(root, size=None, is_mask=False):
	raw = [util.read_raw_bytes(name, root=root) for name in os.listdir(root)]
	if size is None:
		return raw
	
	images = []
	for data in raw:
		image = util.str_to_byte_img(data) if is_mask else util.str_to_rgb(data)
		image = cv2.resize(image, (size,size))
		images.append(image)
	return images
	
@fig.Script('download-mvtec', description='Download and Format MVTec Anomaly Detection dataset')
def download_mvtec(A):
	
	root = A.pull('root', os.environ.get('FOUNDATION_DATA_DIR', None))
	
	if root is None:
		raise Exception('Must specify a FOUNDATION_DATA_DIR as env variable (or pass in a `root`)')
	
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

@Dataset('mvtec')
class MVTec_Anomaly_Detection(Info_Dataset):
	
	def __init__(self, A):
	
		dataroot = A.pull('dataroot')
		
		size = A.pull('size', None)
		
		dirname = 'full' if size is None else f'size{size}'
		droot = os.path.join(dataroot, dirname)
		
		if not os.path.isdir(dataroot) or not os.path.isdir(droot):
			download = A.pull('download', False)
			if not download:
				raise Exception('This dataset hasnt been downloaded and setup yet '
				                '(set the "download" argument to do so automatically)')
		
			print(f'Downloading/Formatting MVTec dataset: {dirname}')
			droot = fig.quick_run('download-mvtec', root=dataroot, size=size)
		
		cats = A.pull('cats', None)
		
		din = (3, 1024, 1024) if size is None else (3, size, size)
		dout = din
		super().__init__(din, dout)
		
		fnames = [c for c in os.listdir(droot) if '.h5' in c and (cats is None or c.split('.')[0] in cats)]
		
		self.files = [hf.File(os.path.join(droot, fname), 'r') for fname in fnames]
		self.size = size
	
		self.images = None
	
		if size is None:
			pass






