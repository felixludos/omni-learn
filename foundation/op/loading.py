
import sys, os
import numpy as np
import torch
import yaml

import omnifig as fig

from .. import util
from .data import get_loaders

FD_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(FD_PATH),'trained_nets')

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
				yaml.dump(data, f)
				
		mpath = os.path.join(root, f'ckpt-model_{steps}.pth.tar')
		torch.save(data, mpath)
	
	if is_best:
		if records is not None:
			rpath = os.path.join(root, f'ckpt-records_best.yaml')
			with open(rpath, 'w') as f:
				yaml.dump(data, f)
		
		mpath = os.path.join(root, f'ckpt-model_best.pth.tar')
		torch.save(data, mpath)


def wrap_datasets(*datasets, A=None):
	num_workers = A.pull('dataset.num_workers', 0)
	batch_size = A.pull('dataset.batch_size', 64)
	shuffle = A.pull('dataset.shuffle', True)
	drop_last = A.pull('dataset.drop_last', True)
	
	return get_loaders(*datasets, batch_size=batch_size, num_workers=num_workers,
	                   shuffle=shuffle, drop_last=drop_last)


def wrap_transaction(create_fn, A, **kwargs):
	A.begin()
	
	obj = create_fn(A, **kwargs)
	
	store_keys = A.pull('store_keys', [])
	
	stored = {}
	for key in store_keys:
		try:
			val = A.pull(key, silent=True)
		except fig.MissingConfigError:
			pass
		else:
			stored[key] = val
	
	A.abort()
	
	for key, val in stored.items():
		A.push(key, val, silent=True, force_root=True)
	
	return obj

def _get_raw_path(A):
	path = A.pull('resume', None)
	loading = False
	if path is None:
		path = A.pull('load', None)
		loading = True
	return path, loading


def find_config(ident):
	path = ident
	if os.path.isfile(ident):
		if 'config.yaml' in ident:
			return ident
		path = os.path.dirname(ident)
	
	if os.path.isdir(path):
		for fname in os.listdir(path):
			if fname == 'config.yaml':
				return os.path.join(path, fname)
	
	raise Exception(f'no config found: {ident}')


def _get_ckpt_num(ident):
	ident = os.path.basename(ident)
	
	terms = ident.split('.')
	if len(terms) == 3:
		name, ext, end = terms
		if name.startswith('checkpoint'):
			return int(name.split('_')[-1])


def _find_ckpt_with(ident, last=False, req='pth.tar'):
	valid = lambda n: ((last and 'checkpoint' in n) or (not last and 'best' in n)) \
	                  and req in n
	path = ident
	
	if os.path.isfile(ident):
		if valid(ident):
			return ident
		path = os.path.dirname(ident)
	
	if os.path.isdir(path):
		names = [fname for fname in os.listdir(path) if valid(path)]
		if len(names) == 1:
			return os.path.join(path, names[0])
		elif len(names) > 1:
			nums = [_get_ckpt_num(fname) for fname in os.listdir(path) if valid(path)]
			
			name = max(zip(names, nums), key=lambda x: x[1] if x[1] is not None else -1)[0]
			return os.path.join(path, name)
	
	raise Exception(f'no checkpoint found: {ident}')


def find_checkpoint(ident, last=False):
	return _find_ckpt_with(ident, last=last, req='pth.tar')


def find_records(ident, last=False):
	return _find_ckpt_with(ident, last=last, req='yaml')


def setup_records(A):
	records = {
		'total_samples': {'train': 0, 'val': 0, 'test': 0, },
		'total_steps': 0,
		'stats': {'train': [], 'val': []},
		
		'epoch': 0,
		'batch': 0,
		'checkpoint': 0,
		'validations': 0,
		
		'epoch_seed': util.gen_deterministic_seed(A.pull('seed')),
	}
	
	track_best = A.pull('training.track_best', False)
	
	if track_best:
		records['best'] = {'loss': None, 'checkpoint': None}
	
	# tau = info.pull('stats_decay', 0.001)
	# util.set_default_tau(tau)
	
	return records


def setup_logging(A):
	invisible = A.pull('output.invisible', False)
	if invisible:
		print('No record of this run will be made')
		return None
	
	name = A.pull('output.name')
	if 'save_dir' not in A.output:
		now = util.get_now()
		logdate = A.pull('output.logdate', True)
		if logdate and '_logged_date' not in A.output:
			name = A.push('output.name', '{}_{}'.format(name, now))
			A.push('output._logged_date', now)
		
		saveroot = A.pull('output.saveroot', None)
		if saveroot is None:
			saveroot = os.environ['FOUNDATION_SAVE_DIR'] \
				if 'FOUNDATION_SAVE_DIR' in os.environ else DEFAULT_SAVE_PATH
			print(f'Default save path chosen: {saveroot}')
		save_dir = A.push('output.save_dir', os.path.join(saveroot, name))
	else:
		save_dir = A.pull('output.save_dir')
	
	tblog = A.pull('output.tblog', False)
	txtlog = A.pull('output.txtlog', False)
	
	if tblog or txtlog:
		util.create_dir(save_dir)
		logtypes = []
		if txtlog:
			logtypes.append('stdout')
		if tblog:
			logtypes.append('on tensorboard')
		print('Logging {}'.format(' and '.join(logtypes)))
	
	logger = util.Logger(save_dir, tensorboard=tblog, txt=txtlog)
	return logger


@fig.Script('load_config')
def load_config(A):
	device = A.push('device', 'cpu', overwrite=not torch.cuda.is_available())
	
	cudnn_det = A.pull('cudnn_det', False)
	if 'cuda' in device and cudnn_det:
		torch.backends.cudnn.deterministic = cudnn_det
	
	A.push('seed', util.gen_random_seed(), overwrite=False, silent=True)
	
	
	# override = A.override if 'override' in A else {}
	override = A.pull('override', {})
	
	extend = A.pull('extend', None)
	
	path, loading = _get_raw_path(A)
	if path is None:
		return A
	
	print(f'Loading Config: {path}')
	
	path = find_config(path)
	
	load_A = fig.get_config(path)
	if loading:
		load_A.update(A)
	else:
		A = load_A
	A.update(override)
	
	if extend is not None:
		A.push('training.step_limit', extend)
	
	return A


@fig.Script('load_records')
def load_records(A):
	
	last = A.pull('last', False)
	
	path = A.pull('path', None)
	loading = False
	
	if path is None:
		path, loading = _get_raw_path(A)
	if path is None or loading:
		return setup_records(A)
	
	print(f'Loading Records: {path}')
	
	path = find_records(path, last=last)
	
	with open(path, 'r') as f:
		records = yaml.safe_load(f)
	return records

