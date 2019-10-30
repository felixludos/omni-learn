
import sys, os, time, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import configargparse

import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt

import foundation as fd
from foundation import models
from foundation import util
from foundation import train
from foundation import data
from foundation.models import unsup

import common

# TODO: define models here

class GAN(unsup.Generative_Adversarial_Network):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.stats.new('disc-verdict-real', 'disc-verdict-fake', 'disc-pretend')

	def _visualize(self, info, logger):
		info.detach()

		if self._viz_counter % 5 == 0:
			logger.add('histogram', 'disc-verdict-real-hist', info.verdict_r.squeeze())
			logger.add('histogram', 'disc-verdict-fake-hist', info.verdict_f.squeeze())
			if 'pretend' in info:
				logger.add('histogram', 'disc-pretend-hist', info.pretend.squeeze())

			B, C, H, W = info.gen.shape
			N = min(B, 32)

			viz_gen = info.gen[:N]
			logger.add('images', 'gen', viz_gen)

		self.stats.update('disc-verdict-real', info.verdict_r.mean())
		self.stats.update('disc-verdict-fake', info.verdict_f.mean())
		if 'pretend' in info:
			self.stats.update('disc-pretend', info.pretend.mean())

def get_options():
	parser = train.get_parser()

	parser.add_argument('--disc-steps', type=int, default=1)
	parser.add_argument('--disc-clip', type=float, default=None)
	parser.add_argument('--disc-gp', type=float, default=None)

	parser.add_argument('--disc-fc', type=int, nargs='+')

	return parser


def get_model(args):

	if args.model_type == 'gan':
		discriminator = models.Conv_Encoder(args.din, latent_dim=1,

		                              nonlin=args.nonlin, output_nonlin=None,

		                              channels=args.channels, kernels=args.kernels,
		                              strides=args.strides, factors=args.factors,
		                              down=args.downsampling, norm_type=args.norm_type,

		                              hidden_fc=args.disc_fc)

		generator = models.Conv_Decoder(args.din, latent_dim=args.latent_dim,

		                              nonlin=args.nonlin, output_nonlin='sigmoid',

		                              channels=args.channels[::-1], kernels=args.kernels[::-1],
		                              ups=args.factors[::-1],
		                              upsampling=args.upsampling, norm_type=args.norm_type,

		                              hidden_fc=args.fc)

		model = GAN(generator=generator, discriminator=discriminator,
		            disc_steps=args.disc_steps, disc_clip=args.disc_clip, disc_gp=args.disc_gp)

		model.set_optim(util.Complex_Optimizer(
			gen=util.get_optimizer(args.optim_type, generator.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
			disc=util.get_optimizer(args.optim_type, discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
		))

		return model




if __name__ == '__main__':

	argv = None
	# argv = ['--config', 'config/gan.yaml', '--name', 'test-wgan', '--no-cuda']

	try:
		train.run_full(get_options, common.get_data, get_model, argv=argv)

	except KeyboardInterrupt:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()

	except:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()
		ipdb.post_mortem(tb)




