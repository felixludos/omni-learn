
import sys, os  #, traceback, ipdb
from tqdm import tqdm

import h5py as hf

import numpy as np

from omnibelt import load_yaml, save_yaml
import omnifig as fig

import torch

from ... import util
# from ... import Generative
from .fid import compute_inception_stat, load_inception_model

@fig.Script('compute-fid', description='Compute an FID stat for a dataset or model')
def compute_fid(A):

	device = A.push('device', 'cuda' if torch.cuda.is_available() else 'cpu', overwrite=False)

	fid_dim = A.pull('fid-dim', '<>dim', 2048)
	assert fid_dim in {64, 192, 768, 2048}, f'invalid dim: {fid_dim}'

	n_samples = A.pull('n-samples', '<>n_samples', 50000)
	
	overwrite = A.pull('overwrite', False)

	use_model = A.pull('use-model', False)
	
	batch_size = None
	run = None
	gen_fn = None
	stats_save_name = None
	
	if use_model:
		raise NotImplementedError # TODO
		
		run = fig.run('load-run', A)
		model = run.get_model()
		
		if not isinstance(model, Generative):
			if A.pull('use-forward', False):
				print(f'Using forward function of model (since its not an instance of Generative)')
				gen_fn = model
			else:
				raise Exception('Model is not compatible - it should use the Generative mixin')
		else:
			gen_fn = model.generate
			
		stats_save_name = A.pull('stats-name', 'fid-stats')
		if run.has_results(stats_save_name) and not overwrite:
			print('Already found fid stats for this model stored (force with "overwrite")')
			data = run.get_results(stats_save_name)
			return data.get('fid-mu', None), data.get('fid-sigma', None)
	
	if run is None:
		print('Computing FID stats for a dataset')

		mode = A.push('mode', 'train', overwrite=False)
	
		dataset = fig.run('load-data', A.sub('dataset'))
		
	
		out_path = A.pull('save-path', '<>out-path', '<>out', None)
	
		if out_path is None:
			ident = A.pull('ident', None)
			ident = 'fid_stats.h5' if ident is None else f'{ident}_fid_stats.h5'
			
			try:
				dataroot = dataset.root
			except:
				dataroot = util.get_data_dir(A)
			name = A.pull('name', '<>dataset.name', dataset.__class__.__name__)
			if '.h5' not in name:
				name = dataroot / ident
			out_path = dataroot / name
	
		print(f'Will save to {out_path}')
		
		f = hf.File(out_path, 'r+') if os.path.isfile(out_path) else hf.File(out_path, 'w')
		
		key = f'{mode}_{fid_dim}'
		
		if f'{key}_mu' in f.keys():
			print(f'Found previously computed stats for {mode} with dim {fid_dim}')
			if not overwrite:
				print('Will not overwrite')
				return f[f'{key}_mu'][()], f[f'{key}_sigma'][()]
	
		if len(dataset) < n_samples:
			print(f'WARNING: dataset only contains {len(dataset)}, so that is set to n-samples')
			n_samples = len(dataset)
	
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

		gen_fn = true_fn

		print('Computing dataset (gt) fid stats')
	
	print('Loading inception model...', end='')
	inception_model = load_inception_model(dim=fid_dim, device=device)
	print('done')

	if batch_size is None:
		batch_size = A.pull('batch_size', 64)
	
	pbar = tqdm if A.pull('pbar', True) else None
	
	m, s = compute_inception_stat(
		gen_fn, inception=inception_model,
	    batch_size=batch_size, n_samples=n_samples,
	    pbar=pbar
	)

	if model is None:
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
	elif stats_save_name is not None:
		run.update_results(stats_save_name, {'mu':m, 'sigma':s}, overwrite=overwrite)

	return m, s



