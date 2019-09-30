
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib

from . import nets
from .. import framework as fm
from .atom import *

class Variational_Autoencoder(nets.Autoencoder): # beta-VAE

	def __init__(self, shape, latent_dim, beta=1., hidden_fc=[], nonlin='prelu', **kwargs):
		super().__init__(shape=shape, latent_dim=latent_dim*2, hidden_fc=hidden_fc, nonlin=nonlin, **kwargs)
		
		self.full_latent_dim = self.latent_dim
		self.latent_dim //= 2
		
		self.dec.fc = make_MLP(self.latent_dim, int(np.product(self.dec.deconv_shape)),
		                        hidden_dims=hidden_fc[::-1], nonlin=nonlin, output_nonlin=nonlin)
		
		self.beta = beta

	def forward(self, x, ret_q=False):

		q = self.get_distrib(x).rsample()
		rec = self.decode(q)

		if ret_q:
			return rec, q
		return rec

	def generate(self, seed=None, N=1):

		if seed is None:
			seed = torch.randn(N, self.latent_dim).to(self.device)
		return self.decode(seed)

	def get_distrib(self, x):

		q = self.encode(x)

		mu, sigma = q.narrow(-1, 0, self.latent_dim), q.narrow(-1, self.latent_dim, self.latent_dim).exp()

		return distrib.Normal(loc=mu, scale=sigma)

	def get_loss(self, x, stats=None):

		p = self.get_distrib(x)

		rec = self.decode(p.loc)

		#rec = self.decode(p.rsample())
		rec_loss = self.criterion(rec, x)

		var_loss = distrib.kl_divergence(p, None).mean()

		if stats is not None:
			if 'rec-loss' not in stats:
				stats.new('rec-loss', 'var-loss')
			stats.update('rec-loss', rec_loss.detach())
			stats.update('var-loss', var_loss.detach())

		#print(self.beta)
		#quit()

		return rec_loss + self.beta*var_loss




