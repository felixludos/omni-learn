
import sys, os, time
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import util
# from ..models import unsup
from .. import models


_model_registry = {}
def register_model(name, cls):
	_model_registry[name] = cls


def fill_args(cls, A):

	args = inspect.getargspec()


def default_create_model(A):
	'''
	req: A.model, A.device, A.model.type


	:param A:
	:return:
	'''

	info = A.model

	assert A.model.type in _model_registry, 'Unknown model type (have you registered it?): {}'.format(A.model.type)


def save_checkpoint(info, save_dir, is_best=False, epoch=None):
	path = None
	if is_best:
		path = os.path.join(save_dir, 'best.pth.tar')
		torch.save(info, path)

	if epoch is not None:
		path = os.path.join(save_dir, 'checkpoint_{}.pth.tar'.format(epoch))
		torch.save(info, path)

	return path

def load(path=None, args=None, load_model=None, load_data=None, load_state_dict=True,
         load_last=False, **overrides):
	assert path is not None or args is not None, 'must provide either path to checkpoint or args'
	assert load_model or load_data, 'nothing to load'

	checkpoint = None
	if path is not None:
		ckptpath = path
		if os.path.isdir(path):
			pick = 'best.pth.tar'
			if load_last:
				ckpts = [n for n in os.listdir(path) if 'checkpoint' in n and '.pth.tar' in n]
				vals = [int(n.split('_')[-1].split('.')[0]) for n in ckpts]
				pick = 'checkpoint_{}.pth.tar'.format(max(vals))
			print('Found {} checkpoints. Using {}'.format(len(ckpts), pick))

			ckptpath = os.path.join(path, pick)

		# if not os.path.isfile(ckptpath):
		# 	ckptpath = os.path.join(path, list(sorted(os.listdir(path), reverse=True))[0], 'best.pth.tar')

		assert os.path.isfile(ckptpath), 'Could not find checkpoint: ' + path

		checkpoint = torch.load(ckptpath)
		args = checkpoint['args']
		print('Loaded {}'.format(ckptpath))

		for k,v in overrides.items():
			args.__dict__[k] = v


	out = []

	if load_data is not None:
		if checkpoint is not None and 'datasets' in checkpoint:
			datasets = checkpoint['datasets']
		else:
			datasets = load_data(args=args)

		out.append(datasets)

	if load_model:

		model = load_model(args=args)

		if checkpoint is not None and 'model_state' in checkpoint and load_state_dict:
			model.load_state_dict(checkpoint['model_state'])
			print('Loaded model_state from checkpoint')

		out.append(model)

	if checkpoint is not None:
		return args, out

	return out






# Deprecated

def load_unsup_model(path=None, args=None, optim=None, to_device=True):
	assert path is not None or args is not None, 'must specify the model'

	checkpoint = None
	if path is not None:
		ckptpath = path
		if os.path.isdir(path):
			ckptpath = os.path.join(path, 'best.pth.tar')

		if not os.path.isfile(ckptpath):
			ckptpath = os.path.join(path, list(sorted(os.listdir(path),reverse=True))[0], 'best.pth.tar')

		assert os.path.isfile(ckptpath), 'Could not find encoder: ' + path

		checkpoint = torch.load(ckptpath)
		args = checkpoint['args']
		print('Loaded {}'.format(path))

	model = None

	criterion = models.get_loss_type(args.loss_type)

	if args.model_type == 'auto':
		model = models.Autoencoder(args.din, latent_dim=args.latent_dim, nonlin=args.nonlin,
								 latent_nonlin=None, recon_nonlin='sigmoid',
								 channels=args.channels, kernels=args.kernels, factors=args.factors,
								 down=args.downsampling, up=args.upsampling, batch_norm=args.batch_norm,
								 hidden_fc=args.fc, criterion=criterion)

	elif args.model_type == 'var':
		model = unsup.Variational_Autoencoder(shape=args.din, latent_dim=args.latent_dim, nonlin=args.nonlin,
								 latent_nonlin=None, recon_nonlin='sigmoid',
								 channels=args.channels, kernels=args.kernels, factors=args.factors,
								 down=args.downsampling, up=args.upsampling, batch_norm=args.batch_norm,
								 hidden_fc=args.fc, criterion=criterion, beta=args.beta)

	else:
		raise Exception('Unknown model config_tml: {}'.format(args.model_type))

	if checkpoint is not None:

		# print(model)

		model.load_state_dict(checkpoint['model_state'])

		model.load_args = args

		try:

			if optim is not None:
				optim.load_state_dict(checkpoint['optim_state'])

		except:
			print('Failed to load optim')

		print('Saved params loaded')

	if to_device:
		model.to(args.device)

	return model


