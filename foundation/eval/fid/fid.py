
import sys, os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import omnifig as fig

from ... import util
from . import utils


def load_inception_model(dim=2048, device='cuda'): # 64, 192, 768, 2048

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



@fig.Component('fid')
class ComputeFID(util.Deviced):
	def __init__(self, A, dim=None, ret_stats=None,
	             batch_size=None, n_samples=None, **kwargs):
		
		if dim is None:
			dim = A.pull('dim', 2048)
		
		skip_load = A.pull('skip_load', False)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 50)
			
		if n_samples is None:
			n_samples = A.pull('n_samples', 50000)
		
		if ret_stats is None:
			ret_stats = A.pull('ret_stats', True)
		
		pbar = A.pull('pbar', None) # TODO: add progress bar for fid computation
		
		super().__init__(A, **kwargs)
	
		self.batch_size = batch_size
		self.n_samples = n_samples
		self.ret_stats = ret_stats
		self.dim = dim
	
		self.pbar = pbar
	
		self.inception = None
		if not skip_load:
			self._load_inception()
		
		self.baseline_stats = None
	
	
	def _load_inception(self, force=False):
		if self.inception is None or force:
			print(f'Loading inception model dim={self.dim}')
			self.inception = load_inception_model(self.dim, self.get_device())
	
	
	def set_baseline_stats(self, stats=None):
		self.baseline_stats = stats
	
	
	def compute_stats(self, generate, batch_size=None, n_samples=None, name=None):
		
		self._load_inception()
		
		if batch_size is None:
			batch_size = self.batch_size
		if n_samples is None:
			n_samples = self.n_samples
		
		stats = compute_inception_stat(generate, inception=self.inception, pbar=self.pbar,
		                               batch_size=batch_size, n_samples=n_samples, name=name)
		
		return stats
	
	
	def compute_distance(self, stats1, stats2=None):
		
		if stats2 is None:
			stats2 = self.baseline_stats
			
		assert stats2 is not None, 'no base stats found'
		return compute_frechet_distance(*stats1, *stats2)



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

