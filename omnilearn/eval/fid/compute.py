
import sys, os  #, traceback, ipdb
from tqdm import tqdm

import h5py as hf

import numpy as np

from omnibelt import load_yaml, save_yaml
import omnifig as fig

import torch

from ... import util
from .fid import compute_inception_stat, load_inception_model

@fig.Script('compute-fid', description='Compute an FID stat for a dataset')
def compute_fid(A):

	device = A.push('device', 'cuda' if torch.cuda.is_available() else 'cpu', overwrite=False)

	fid_dim = A.pull('fid-dim', '<>dim', 2048)
	assert fid_dim in {64, 192, 768, 2048}, f'invalid dim: {fid_dim}'

	n_samples = A.pull('n-samples', '<>n_samples', 50000)

	mode = A.push('mode', 'train', overwrite=False)

	dataset = fig.run('load-data', A.sub('dataset'))

	out_path = A.pull('save-path', '<>out-path', '<>out', None)

	if out_path is None:
		ident = A.pull('ident', None)
		ident = 'fid_stats.h5' if ident is None else f'{ident}_fid_stats.h5'
		
		dataroot = A.pull('root', '<>dataset.dataroot', None)
		name = A.pull('name', '<>dataset.name', dataset.__class__.__name__)
		if '.h5' not in name:
			name = os.path.join(name, ident)
		out_path = os.path.join(dataroot, name)

	print(f'Will save to {out_path}')
	
	f = hf.File(out_path, 'r+') if os.path.isfile(out_path) else hf.File(out_path, 'w')
	
	key = f'{mode}_{fid_dim}'
	
	if f'{key}_mu' in f.keys():
		print(f'Found previously computed stats for {mode} with dim {fid_dim}')
		overwrite = A.pull('overwrite', False)
		if not overwrite:
			print('Will not overwrite')
			return f[f'{key}_mu'][()], f[f'{key}_sigma'][()]

	if len(dataset) < n_samples:
		print(f'WARNING: dataset only contains {len(dataset)}, so that is set to n-samples')
		n_samples = len(dataset)

	pbar = tqdm if A.pull('pbar', True) else None

	print('Loading inception model...', end='')
	inception_model = load_inception_model(dim=fid_dim, device=device)
	print('done')

	loader = dataset.get_loader(infinite=True)
	
	def true_fn(N):
		
		batch = loader.demand(N)

		if isinstance(batch, torch.Tensor):
			imgs = batch
		elif isinstance(batch, (list, tuple)):
			imgs = batch[0]
		elif isinstance(batch, dict):
			imgs = batch['x']
			
		return imgs.to(device)
	
	batch_size = dataset.get_batch_size()

	print('Computing dataset (gt) fid stats')

	m, s = compute_inception_stat(
		true_fn, inception=inception_model,
	    batch_size=batch_size, n_samples=n_samples,
	    pbar=pbar
	)

	print('Dataset (gt) fid stats computed.')

	mkey = f'{mode}_{fid_dim}_mu'
	if mkey in f:
		f[mkey][:] = m
	else:
		f.create_dataset(mkey, data=m)

	skey = f'{mode}_{fid_dim}_sigma'
	if skey in f:
		f[skey][:] = s
	else:
		f.create_dataset(skey, data=s)

	f.close()

	return m, s



