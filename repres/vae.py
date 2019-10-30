
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
plt.switch_backend('Agg')

from sklearn.decomposition import PCA

import foundation as fd
from foundation import models
from foundation.models.unsup import Variational_Autoencoder
from foundation import util
from foundation import train
from foundation import data

import common


class VAE(Variational_Autoencoder): # already defined in foundation


	def _visualize(self, info, logger):
		if self._viz_counter % 5 == 0:
			# logger.add('histogram', 'latent-norm', info.latent.norm(p=2, dim=-1))

			logger.add('histogram', 'latent-mu', info.latent.loc.contiguous().view(-1))
			logger.add('histogram', 'latent-sigma', info.latent.scale.contiguous().view(-1))

			B, C, H, W = info.original.shape
			N = min(B, 8)

			viz_x, viz_rec = info.original[:N], info.reconstruction[:N]

			recs = torch.cat([viz_x, viz_rec], 0)
			logger.add('images', 'rec', recs)

			# show some generation examples
			try:
				gen = self.generate(2 * N)
				logger.add('images', 'gen', gen)
			except NotImplementedError:
				pass

			# show latent space
			if self.latent_dim >= 2:
				info.latent = info.latent.sample()
				if self.latent_dim > 2:
					x = PCA(n_components=2, copy=False).fit_transform(info.latent.cpu().numpy())
				else:
					x = info.latent.cpu().numpy()

				fig = plt.figure(figsize=(3, 3))
				plt.gca().set_aspect('equal')
				plt.scatter(*x.T, marker='.', s=6, edgecolors='none', alpha=.7)
				# plt.show()
				logger.add('figure', 'latent-space', fig, close=True)

		if 'reg' in info:
			self.stats.update('reg', info.reg)


def get_options():
	parser = train.get_parser()

	parser.add_argument('--beta', type=float, default=None)
	parser.add_argument('--criterion', type=str, default='bce')

	return parser


def get_model(args):

	if args.model_type == 'vae':

		encoder = models.Conv_Encoder(args.din, latent_dim=args.latent_dim * 2,

		                              nonlin=args.nonlin, output_nonlin=None,

		                              channels=args.channels, kernels=args.kernels,
		                              strides=args.strides, factors=args.factors,
		                              down=args.downsampling, norm_type=args.norm_type,

		                              hidden_fc=args.fc)


		decoder = models.Conv_Decoder(args.din, latent_dim=args.latent_dim,

		                              nonlin=args.nonlin, output_nonlin='sigmoid',

		                              channels=args.channels[::-1], kernels=args.kernels[::-1],
		                              ups=args.factors[::-1],
		                              upsampling=args.upsampling, norm_type=args.norm_type,

		                              hidden_fc=args.fc[::-1])

		model = VAE(encoder=encoder, decoder=decoder, beta=args.beta, criterion=util.get_loss_type(args.criterion, reduction='sum'))

		model.set_optim(optim_type=args.optim_type, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

		return model

	raise NotImplementedError




if __name__ == '__main__':

	argv = None

	try:
		train.run_full(get_options, common.get_data, get_model, argv=argv)

	except KeyboardInterrupt:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()

	except:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()
		ipdb.post_mortem(tb)




