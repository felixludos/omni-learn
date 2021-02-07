

import sys, os
from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import requests

from omnibelt import get_printer

prt = get_printer(__name__)

OMNILEARN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# _warn_savedir = True
# def get_save_dir(A=None):
#
# 	if A is not None:
# 		root = A.pull('saveroot', None)
# 		if root is not None:
# 			return Path(root)
#
# 	if 'OMNILEARN_SAVE_DIR' in os.environ:
# 		return Path(os.environ['OMNILEARN_SAVE_DIR'])
#
# 	root = Path(os.getcwd())
# 	root = root / 'trained_nets'
# 	root.mkdir(exist_ok=True)
#
# 	global _warn_savedir
# 	if _warn_savedir:
# 		prt.warning(f'No savedir found (specify with "OMNILEARN_SAVE_DIR" env variable), '
# 		            f'now using {str(root)}')
# 		_warn_savedir = False
#
# 	return root

_warn_datadir = True
def get_data_dir(A=None, silent=False):
	
	if A is not None:
		root = A.pull('dataroot', None, silent=silent)
		if root is not None:
			return Path(root)
		
	if 'OMNILEARN_DATA_DIR' in os.environ:
		return Path(os.environ['OMNILEARN_DATA_DIR'])
	
	root = Path(os.getcwd())
	root = root / 'local_data'
	root.mkdir(exist_ok=True)
	
	global _warn_datadir
	if _warn_datadir:
		prt.warning(f'No datadir found (specify with "OMNILEARN_DATA_DIR" env variable), '
					f'now using {str(root)}')
		_warn_datadir = False
	
	return root


def save_figure(name, fg=None, root=None, force=False, exts=None, **kwargs):
	
	if exts is None:
		exts = ['png']
	
	if fg is None:
		fg = plt.gcf()
	
	if root is not None or force:
		
		if root is None:
			path = name
		else:
			path = str(Path(root) / name)
		
		for ext in exts:
			fg.savefig(path + f'.{ext}', **kwargs)
		
		print(f'Figure {name} saved as {set(exts)}')
	


def get_patch(path, ht=None, wd=None, ret_bgr=False):

	img = cv2.imread(path)

	if not ret_bgr:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # COLOR_BGR2HSV

	H, W = img.shape[0], img.shape[1]

	if ht is None:
		ht = H
	if wd is None:
		wd = W

	i = np.random.randint(H - ht) if H > ht else 0
	j = np.random.randint(W - wd) if W > wd else 0
	patch = img[i:i+ht, j:j+wd]
	if patch.shape[:2] != (ht, wd):
		patch = cv2.resize(patch, (ht, wd))
	return patch

def get_img(path, ht=None, wd=None, ret_bgr=False, pad=False):
	if ht is not None and wd is None:
		wd = ht

	img = cv2.imread(path)
	H, W = img.shape[:2]

	if not ret_bgr:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	if ht is None or (ht, wd) == (H,W):
		return img
	elif pad and ht is not None and H < ht:
		pimg = np.zeros((ht,wd,3), dtype=np.uint8)
		pH, pW = (ht-H)//2, (wd-W)//2
		pimg[pH:pH+H, pW:pW+W] = img
		return pimg

	return cv2.resize(img, (wd, ht))



def download_file_from_google_drive(ID, out_dir, name=None, pbar=None, chunk_size=32768):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : ID}, stream = True)
	token = get_confirm_token(response)

	if token is None:
		response.raise_for_status()

	params = { 'id' : ID, 'confirm' : token}
	response = session.get(URL, params = params, stream = True)

	out_dir = Path(out_dir)
	out_dir.mkdir(exist_ok=True)
	
	if name is None:
		name = response.headers.get('Content-Disposition', '"download').split('"')[1]
	
	idx = 1
	if name is None:
		name = 'download'
	_name = Path(name)
	base, ext = _name.stem, _name.suffix
	while (out_dir / name).exists():
		name = f'{base}_{idx}.{ext}' if len(ext) else f'{base}_{idx}'
		idx += 1
	
	dest = out_dir / name
	
	save_response_content(response, dest, chunk_size=chunk_size, pbar=pbar)
	
	return dest

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination, chunk_size=32768, pbar=None):
	
	total_size_in_bytes = int(response.headers.get('content-length', 0))
	if pbar is not None:
		progress_bar = pbar(total=total_size_in_bytes, unit='iB', unit_scale=True)
		progress_bar.set_description(f'Downloading {destination.name}')

	with open(str(destination), "wb") as f:
		for chunk in response.iter_content(chunk_size):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)
				if pbar is not None:
					progress_bar.update(len(chunk))
	
	if pbar is not None:
		progress_bar.close()




