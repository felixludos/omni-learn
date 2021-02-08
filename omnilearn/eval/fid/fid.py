
import sys, os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnibelt import get_printer
import omnifig as fig

from ... import util
from . import utils

prt = get_printer(__name__)

def load_inception_model(dim=2048, device='cuda'): # 64, 192, 768, 2048

	if not torch.cuda.is_available(): # cuda not found
		prt.warning('cuda not found')
		device = 'cpu'
	if device == 'cpu':
		prt.warning('using cpu for inception model - this will take a long time!')

	block_idx = utils.InceptionV3.BLOCK_INDEX_BY_DIM[dim]

	model = utils.InceptionV3([block_idx]).to(device).eval()
	model._dim = dim
	model._device = device
	return model


def apply_inception(samples, inception, include_grad=False):
	N, C, H, W = samples.shape
	if C == 1:
		samples = torch.cat([samples] * 3, dim=1)
	if C > 3:
		samples = samples[:, :3].contiguous()

	if include_grad:
		pred = inception(samples)[0]
	else:
		with torch.no_grad():
			pred = inception(samples)[0]

	if pred.shape[2] != 1 or pred.shape[3] != 1:
		pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

	return pred



def compute_inception_stat(generate, inception=None, batch_size=50, n_samples=50000,
                     dim=2048, device='cuda', pbar=None, name=None):
	"""

	:param generate: callable: input N (int) -> (N, 3, H, W) images (as pytorch tensor)
	:param inception: inception model (if None will automatically load)
	:param batch_size: int
	:param n_samples: int
	:param pbar: tqdm-like (optional)
	:param dim: dimension for inception model (only used if inception is not provided)
	:param device: device for inception model (only used if inception is not provided)
	"""

	if inception is not None:
		dim = inception._dim
		device = inception._device

	pred_arr = np.empty((n_samples, dim)) # TODO: refactor to keep incremental statistics to avoid storing the full set

	if inception is None:
		inception = load_inception_model(dim=dim, device=device)
	
	title = f' {name}' if name is not None else ''
	title = f'Computing{title} FID'

	if pbar is not None:
		pbar = pbar(total=n_samples)
		pbar.set_description(title)
	elif name is not None:
		print(title)

	j = 0
	while j < n_samples:
		N = min(batch_size, n_samples - j)

		samples = generate(N)

		pred = apply_inception(samples, inception)

		pred_arr[j:j+N] = pred.cpu().numpy().reshape(N, -1)

		j += N
		if pbar is not None:
			pbar.update(N)

	if pbar is not None:
		pbar.close()


	m = np.mean(pred_arr, axis=0)
	s = np.cov(pred_arr, rowvar=False)

	return m, s


def compute_frechet_distance(m1, s1, m2, s2, eps=1e-6):
	return utils.calculate_frechet_distance(m1,s1,m2,s2, eps=eps)

