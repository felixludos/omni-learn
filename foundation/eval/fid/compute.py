
import sys, os  #, traceback, ipdb
from tqdm import tqdm

import numpy as np

from omnibelt import load_yaml, save_yaml
import omnifig as fig

import torch

from ... import util
from .fid import compute_inception_stat, load_inception_model

@fig.Script('compute-fid', description='Comppute an FID stat for a dataset')
def compute_fid(A):

	device = A.push('device', 'cuda' if torch.cuda.is_available() else 'cpu', overwrite=False)

	fid_dim = A.pull('fid-dim', 2048)
	assert fid_dim in {64, 192, 768, 2048}, f'invalid dim: {fid_dim}'

	n_samples = A.pull('n-samples', 50000)

	mode = A.push('mode', 'train', overwrite=False)

	dataset = fig.run('load_data', A.sub('dataset'))

	out_path = A.pull('save-path', '<>out-path', '<>out', None)

	if out_path is None:
		dataroot = A.pull('root', '<>dataset.dataroot', None)
		name = A.pull('name', '<>dataset.name', dataset.__class__.__name__)
		if 'yaml' not in name:
			name = os.path.join(name, 'fid_stats.yaml')
		out_path = os.path.join(dataroot, name)

	print(f'Will save to {out_path}')

	if os.path.isfile(out_path):
		print(f'Found existing stats in {out_path}')
		old = load_yaml(out_path)
	else:
		old = []

	stats = []
	for stat in old:
		if stat['mode'] == mode and stat['dim'] == fid_dim:
			print(f'Found previously computed stats for {mode} with dim {fid_dim}')
			overwrite = A.pull('overwrite', False)
			if not overwrite:
				print('Will not overwrite')
				return np.array(stats['mu']), np.array(stats['sigma'])
		else:
			stats.append(stat)

	if len(dataset) < n_samples:
		print(f'WARNING: dataset only contains {len(dataset)}, so that is set to n-samples')
		n_samples = len(dataset)

	pbar = tqdm if A.pull('pbar', True) else None

	print('Loading inception model...', end='')
	inception_model = load_inception_model(dim=fid_dim, device=device)
	print('done')

	loader = dataset.to_loader(A)
	true_loader = util.make_infinite(loader)
	def true_fn(N):
		return util.to(true_loader.demand(N), device)[0]

	batch_size = loader.get_batch_size()

	print('Computing dataset (gt) fid stats')

	m, s = compute_inception_stat(
		true_fn, inception=inception_model,
	    batch_size=batch_size, n_samples=n_samples,
	    pbar=pbar
	)

	print('Dataset (gt) fid stats computed.')

	stats.append({
		'mu': m.tolist(),
		'sigma': s.tolist(),
		'mode': mode,
		'dim': fid_dim,
	})

	save_yaml(stats, out_path)

	return m, s



