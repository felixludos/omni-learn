
import sys, os
import numpy as np
import torch
import yaml

import omnifig as fig

from .. import util
from .runs import Run
# from .data import get_loaders

@fig.AutoModifier('torch')
class Torch(Run):
	def _load_results(self, ident):

		root = self.save_path

		available = set(os.listdir(self.save_path))

		fixed = ident.split('.')[0]
		fname = f'{ident}.pth.tar'

		if ident in available:
			return torch.load(os.path.join(root, ident))

		if fixed in available:
			return torch.load(os.path.join(root, fixed))

		if fname in available:
			return torch.load(os.path.join(root, fname))

		raise FileNotFoundError(f'Unknown ident: {ident}')


def save_checkpoint(root, model, records=None, steps=None, is_best=False):
	
	assert steps is not None or is_best, 'Nothing worth saving'
	
	data = {
		'model_str': str(model),
		'model_state': model.state_dict(),
		
		# 'records': records,
	}

	if steps is not None:
		if records is not None:
			rpath = os.path.join(root, f'ckpt-records_{steps}.yaml')
			with open(rpath, 'w') as f:
				yaml.dump(records, f)
				
		mpath = os.path.join(root, f'ckpt-model_{steps}.pth.tar')
		torch.save(data, mpath)
	
	if is_best:
		if records is not None:
			rpath = os.path.join(root, f'ckpt-records_best.yaml')
			with open(rpath, 'w') as f:
				yaml.dump(records, f)
		
		mpath = os.path.join(root, f'ckpt-model_best.pth.tar')
		torch.save(data, mpath)


# def wrap_datasets(*datasets, A=None):
# 	num_workers = A.pull('dataset.num_workers', 0)
# 	batch_size = A.pull('dataset.batch_size', 64)
# 	shuffle = A.pull('dataset.shuffle', True)
# 	drop_last = A.pull('dataset.drop_last', True)
#
# 	return get_loaders(*datasets, batch_size=batch_size, num_workers=num_workers,
# 	                   shuffle=shuffle, drop_last=drop_last)


def get_raw_path(A):
	path = A.pull('resume', None)
	loading = False
	if path is None:
		path = A.pull('load', None)
		loading = True
	return path, loading


def respect_config(A):
	device = A.push('device', 'cpu', overwrite=not torch.cuda.is_available())
	
	cudnn_det = A.pull('cudnn_det', False)
	if 'cuda' in device and cudnn_det:
		torch.backends.cudnn.deterministic = cudnn_det
	
	A.push('seed', util.gen_random_seed(), overwrite=False, silent=True)
	
#
# @fig.Script('load_config')
# def load_config(A):
#
#
# 	# override = A.override if 'override' in A else {}
# 	override = A.pull('override', {})
#
# 	extend = A.pull('extend', None)
#
# 	raw_path, loading = get_raw_path(A)
# 	if raw_path is None:
# 		return A
#
# 	print(f'Loading Config: {raw_path}')
#
# 	path = find_config(raw_path)
#
# 	load_A = fig.get_config(path)
# 	if loading:
# 		load_A.update(A)
# 	else:
# 		A = load_A
# 	A.update(override)
#
# 	A.push('load' if loading else 'resume', raw_path, overwrite=True)
#
# 	if extend is not None:
# 		A.push('training.step_limit', extend)
#
# 	return A

#
# @fig.Script('load_records')
# def load_records(A):
#
# 	last = A.pull('last', False)
#
# 	path = A.pull('path', None)
# 	loading = False
#
# 	if path is None:
# 		path, loading = get_raw_path(A)
# 	if path is None or loading:
# 		return setup_records(A)
#
# 	print(f'Loading Records: {path}')
#
# 	ckpt_path = find_checkpoint(path, last=last)
# 	A.push('model._load_params', ckpt_path, overwrite=True)
#
# 	path = find_records(path, last=last)
#
# 	with open(path, 'r') as f:
# 		records = yaml.safe_load(f)
# 	return records
#
