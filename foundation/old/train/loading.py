

import sys,os,time
import numpy as np
import torch

from foundation import util

from .config import get_config
from .data import default_load_data#, split_dataset, simple_split_dataset
from .model import default_create_model


def save_checkpoint(info, save_dir, is_best=False, epoch=None):
	path = None
	if is_best:
		path = os.path.join(save_dir, 'best.pth.tar')
		torch.save(info, path)

	if epoch is not None:
		path = os.path.join(save_dir, 'checkpoint_{}.pth.tar'.format(epoch))
		torch.save(info, path)

	return path

def find_checkpoint(path, load_last=False, saveroot=None):
	if os.path.isfile(path) and '.pth.tar' in path:
		return path

	if saveroot is None and 'FOUNDATION_SAVE_DIR' in os.environ:
		saveroot = os.environ['FOUNDATION_SAVE_DIR']

	if not os.path.isdir(path) and saveroot is not None:
		try:
			long_path = os.path.join(saveroot, path)
			if os.path.isfile(long_path):
				return long_path
			assert os.path.isdir(long_path)
		except AssertionError:
			pass
		else:
			path = long_path

	if os.path.isdir(path):
		if not load_last and 'best.pth.tar' in os.listdir(path):
			pick = 'best.pth.tar'
		else:
			ckpts = [n for n in os.listdir(path) if 'checkpoint' in n and '.pth.tar' in n]
			vals = [int(n.split('_')[-1].split('.')[0]) for n in ckpts]
			if len(vals): # dir exists but no checkpoints
				pick = 'checkpoint_{}.pth.tar'.format(max(vals))
				# print('Found {} checkpoints. Using {}'.format(len(ckpts), pick))
			elif 'config.yml' in os.listdir(path):
				# print('Found 0 checkpoints. However, a config file was found')
				return path
		path = os.path.join(path, pick)

		return path

	raise FileNotFoundError(path)



def load(path=None, config=None,
         get_model='default', get_data='default', mode='train',
         update_config=False,
         load_optim=True, load_scheduler=True,
         load_state_dict=True, load_last=False, force_load_model=True,
         strict_load_state=True,
         return_args=False, return_ckpt=False, seed=0, silent=False):
	'''
	Top-level (generally recommended for training/evaluation scripts) function to load models and/or datasets
	
	:param path: name of (or path to) run dir (usually dir in FOUNDATION_DATA_DIR) containing config.yml and checkpoints
	:param config: optional config to use to load if path is not provided
	:param get_model: usually best to leave this as default
	:param get_data: usually best to leave this as default
	:param mode: usually best to leave this as "train"
	:param update_config: if a 'path' and 'config' is provided, this bool determines whether the config should
	be updated with the loaded config
	:param load_optim: usually true unless the optimizer has to be changed
	:param load_scheduler: usually true unless a new scheduler should be used
	:param load_state_dict: load the saved model parameters (not just architecture/hyperparams)
	:param load_last: load last checkpoint found (
	:param force_load_model: throw exception if unable to load a model
	:param return_args:
	:param return_ckpt:
	:param seed: initial random seed (gets reset for model loading and dataset loading)
	:param silent: don't print progress messages
	:return: a list of objects, usually including model and datasets but possibly also the config and loaded checkpoint
	'''
	assert path is not None or config is not None, 'must provide either path to checkpoint or config'
	assert get_model is not None or get_data is not None or return_ckpt, 'nothing to load'

	if get_model is 'default':
		get_model = default_create_model
	if get_data is 'default':
		get_data = default_load_data

	if config is not None and 'load' in config:
		path = config.load
		if not silent:
			print(f'Will attempt to load {path} found in the config')
		del config.load

	checkpoint = None
	if path is not None:
		ckptpath = find_checkpoint(path, load_last=load_last)

		# print(ckptpath)
		# print(os.path.isfile(ckptpath))
		# sys.stdout.flush()

		if os.path.isfile(ckptpath):

			try:
				checkpoint = torch.load(ckptpath)
			except Exception as e:
				print('bad')
				print(e)
				sys.stdout.flush()
				raise e
			run_dir = os.path.dirname(ckptpath)

			if not silent:
				print('load successful')

		elif force_load_model:
			raise FileNotFoundError
		else:
			run_dir = ckptpath
		config_name = 'config.yml' #if 'config.yml' in os.listdir(run_dir) else 'config.tml'
		load_A = get_config(os.path.join(run_dir, config_name))
		if config is None: # if no config is provided, the loaded config is adopted
			config = load_A
		elif update_config:
			new_A = config.copy()
			config.clear()
			config.update(load_A)
			config.update(new_A)
		if not silent:
			print('Loaded {}'.format(ckptpath))

		if 'FOUNDATION_DATA_DIR' in os.environ: # TODO: necessary?
			config.dataroot = os.environ['FOUNDATION_DATA_DIR']
			if not silent:
				print('Set dataroot to: {}'.format(config.dataroot))

	assert config is not None, 'Nothing to get'

	if 'seed' in config:
		seed = config.seed
	elif not silent:
		print('WARNING: no seed found, using: seed={}'.format(seed))

	out = []

	if return_args:
		out.append(config)

	if get_data is not None:
		util.set_seed(seed)

		info = config.dataset

		if checkpoint is not None and 'datasets' in checkpoint:
			datasets = checkpoint['datasets']
		else:
			config.begin()

			dataset = get_data(config.dataset, mode=mode)

			# if get_model is None:
			# 	config.abort() # TODO: don't abort when creating the model right afterwards (wait until after model)
			config.abort()

			try:
				config.din, config.dout = dataset.din, dataset.dout
			except AttributeError as e:
				print('WARNING: Dataset does not have a "din" and "dout"')
				raise e

			trainsets = dataset,
			testset = None
			if 'test_split' in info:
				raise NotImplementedError
				assert 0 < info.test_split < 1, 'cant split: {}'.format(info.val_split)
				*trainsets, testset = simple_split_dataset(dataset, 1-info.test_split, shuffle=True)
			if 'val_split' in info:  # use/create validation set
				# assert 0 < info.val_split < 1, 'cant split: {}'.format(info.val_split)
				trainsets = dataset.split(info)

			datasets = (*trainsets, testset)

		out.append(datasets)


	if get_model is not None:
		util.set_seed(seed)

		info = config.model

		# if get_data is None:
		# 	config.begin()
		config.begin()
		model = get_model(info)
		config.abort() # undo all changes to the config throughout model creation

		if not silent:
			print('Moving model to {}'.format(config.device))
			sys.stdout.flush()

		model.to(config.device)

		if not silent:
			print('Model on {}'.format(config.device))
			sys.stdout.flush()

		if checkpoint is not None and 'model_state' in checkpoint and load_state_dict:

			params = checkpoint['model_state'].copy()

			if 'optim' in params and not load_optim:
				del params['optim']

			if 'scheduler' in params and not load_scheduler:
				del params['scheduler']

			model.load_state_dict(params, strict=strict_load_state)
			if not silent:
				print('Loaded model_state from checkpoint')

		out.append(model)

	if return_ckpt:
		out.append(checkpoint)

	return out

