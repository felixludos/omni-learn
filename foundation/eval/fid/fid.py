
import sys, os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

def load_inception_model(dim=2048, device='cuda'):

	if not torch.cuda.is_available(): # cuda not found
		print('WARNING: cuda not found')
		device = 'cpu'
	if device == 'cpu':
		print('WARNING: running on cpu - this will take a long time!')

	block_idx = utils.InceptionV3.BLOCK_INDEX_BY_DIM[dim]

	model = utils.InceptionV3([block_idx]).to(device).eval()
	model._dim = dim
	model._device = device
	return model



def compute_inception_stat(generate, inception=None, batch_size=50, n_samples=50000,
                     dim=2048, device='cuda', pbar=None):
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

	if pbar is not None:
		pbar = pbar(total=n_samples)

	j = 0
	while j < n_samples:
		N = min(batch_size, n_samples - j)

		samples = generate(N)
		with torch.no_grad():
			pred = inception(samples)[0]

		if pred.shape[2] != 1 or pred.shape[3] != 1:
			pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

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

