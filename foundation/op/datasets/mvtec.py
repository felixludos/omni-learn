
import sys, os

from tqdm import tqdm
try:
	import wget
except ImportError:
	print('WARNING: unable to import wget')
import shutil
import h5py as hf

import numpy as np
import torch

from omnibelt import create_dir
import omnifig as fig

from ... import util
from ..data import Dataset

from ...data import standard_split, Device_Dataset, Info_Dataset, Splitable_Dataset, Testable_Dataset, Batchable_Dataset, Image_Dataset

DATASET_URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'

@fig.Script('download-mvtec', description='Download and Format MVTec Anomaly Detection dataset')
def download_mvtec(A):
	root = A.pull('root', os.environ.get('FOUNDATION_DATA_DIR', None))
	
	if root is None:
		raise Exception('Must specify a FOUNDATION_DATA_DIR as env variable (or pass in a `root`)')
	
	root = os.path.join(root, 'mvtec')
	
	if not os.path.isdir(root):
		create_dir(root)
	
	raw_name = A.pull('raw-name', 'mvtec_anomaly_detection.tar.xz', silent=True)
	rawpath = os.path.join(root, raw_name)
	
	if raw_name not in os.listdir(root):
		print('Downloading MVTec AD dataset')
		
		url = A.pull('url', DATASET_URL)
		
		rawpath = wget.download(url, out=root)
		
	contents = os.listdir(root)
	
	if raw_name in contents and 'readme.txt' not in contents:
		
		print(f'Extracting MVTec AD dataset from {rawpath} file')
	
		shutil.unpack_archive(rawpath, extract_dir=root)
		
		if A.pull('cleanup', True):
			os.remove(rawpath)
	
	todo = [name for name in os.listdir(root)
	              if os.path.isdir(os.path.join(root, name)) and f'{name}.h5' not in contents]
	
	print(f'Formatting {len(todo)} object directories')
	
	if A.pull('pbar', True, silent=True):
		todo = tqdm(todo)
	
	for cat in todo:
		
		with hf.File(f'{cat}.h5', 'w') as f:
		
			croot = os.path.join(root, cat)
		
			mroot = os.path.join(croot, 'ground_truth')
		
			defects = os.listdir(mroot)
			f.attrs['defects'] = defects
		
			for defect in defects:
				droot = os.path.join(mroot, defect)
				f.create_dataset(f'mask_{defect}', data=[util.read_raw_bytes(name, root=droot)
				                                         for name in os.listdir(droot)])
		
			mroot = os.path.join(croot, 'train')
			
			f.create_dataset()
		
		pass
	
	pass

@Dataset('mvtec')
class MVTec_Anomaly_Detection(Info_Dataset):
	
	
	
	def __init__(self):
	
		pass







