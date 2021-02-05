

import sys, os
from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

OMNILEARN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

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


