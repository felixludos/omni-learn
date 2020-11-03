
import sys, os, time
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import omnifig as fig

from . import nets
from .. import framework as fm
from .. import util
from .atom import *

@fig.AutoComponent('ae')
class Autoencoder(fm.Regularizable, fm.Encodable, fm.Decodable, fm.Full_Model):
	def __init__(self, encoder, decoder, reg='L2', reg_wt=0, criterion=None, viz_latent=True, viz_rec=True):

		super().__init__(encoder.din, decoder.dout)

		self.enc = encoder
		self.dec = decoder

		if criterion is None:
			criterion = nn.MSELoss()
		self.criterion = util.get_loss_type(criterion)

		self.reg_wt = reg_wt
		self.reg_fn = util.get_regularization(reg, reduction='sum') \
			if reg_wt is not None and reg_wt > 0 else None

		if self.reg_wt is not None and self.reg_wt > 0:
			self.stats.new('reg')
		self.stats.new('rec_loss')

		self.register_buffer('_q', None, save=True)
		self.register_cache('_real', None)
		self.register_cache('_rec', None)

		self.latent_dim = self.dec.din
		
		self.viz_latent = viz_latent
		self.viz_rec = viz_rec
		
	def _visualize(self, info, logger):
		
		if isinstance(self.enc, fm.Visualizable):
			self.enc.visualize(info, logger)
		if isinstance(self.dec, fm.Visualizable):
			self.dec.visualize(info, logger)
		
		q = None
		if self.viz_latent and 'latent' in info and info.latent is not None:
			q = info.latent.loc if isinstance(info.latent, distrib.Distribution) else info.latent
			
			shape = q.size()
			if len(shape) > 1 and np.product(shape) > 0:
				try:
					logger.add('histogram', 'latent-norm', q.norm(p=2, dim=-1))
					logger.add('histogram', 'latent-std', q.std(dim=0))
					if isinstance(info.latent, distrib.Distribution):
						logger.add('histogram', 'logstd-hist', info.latent.scale.add(1e-5).log().mean(dim=0))
				except ValueError:
					print('\n\n\nWARNING: histogram just failed\n')
					print(q.shape, q.norm(p=2, dim=-1).shape)
		
		B, C, H, W = info.original.shape
		N = min(B, 8)
		
		if self.viz_rec:
			if 'reconstruction' in info:
				viz_x, viz_rec = info.original[:N], info.reconstruction[:N]
				
				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', util.image_size_limiter(recs))
			elif self._rec is not None:
				viz_x, viz_rec = self._real[:N], self._rec[:N]
				
				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', util.image_size_limiter(recs))
		
		logger.flush()
	
	def forward(self, x, ret_q=False):

		q = self.encode(x)
		x = self.decode(q)

		if ret_q:
			return x, q
		return x

	def encode(self, x):
		return self.enc.encode(x)

	def regularize(self, q, p=None):
		B = q.size(0)
		mag = self.reg_fn(q)
		return mag / B

	def decode(self, q):
		return self.dec.decode(q)

	def preprocess(self, x):
		return x

	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		x = batch[0]
		B = x.size(0)

		x = self.preprocess(x)

		out.original = x

		rec, q = self(x, ret_q=True)
		out.latent = q
		out.reconstruction = rec
		
		self._rec, self._real = rec.detach(), x.detach()

		loss = self.criterion(rec, x) / B
		out.rec_loss = loss
		self.stats.update('rec_loss', loss)

		if self.reg_wt > 0:
			reg_loss = self.regularize(q)
			self.stats.update('reg', reg_loss)
			out.reg_loss = reg_loss
			loss += self.reg_wt * reg_loss

		out.loss = loss

		if self.train_me():
			self._q = q.loc.detach() if isinstance(q, distrib.Normal) else q.detach()

			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out


class Variational_Autoencoder(fm.Generative, Autoencoder):

	def __init__(self, *args, min_log_std=None, **kwargs):
		super().__init__(*args, **kwargs)

		assert self.enc.dout//2 == self.dec.din, 'enc/dec not compatible: {} vs {}'.format(self.enc.dout, self.dec.din)

		self.min_log_std = min_log_std

	def decode(self, q):
		if isinstance(q, distrib.Distribution):
			q = q.rsample()
		return super().decode(q)

	def encode(self, x):
		q = super().encode(x)

		mu = q.narrow(-1, 0, self.latent_dim)
		logsigma = q.narrow(-1, self.latent_dim, self.latent_dim)
		if self.min_log_std is not None:
			logsigma = logsigma.clamp(min=self.min_log_std)
		sigma = logsigma.exp()

		return distrib.Normal(loc=mu, scale=sigma)

	# def _visualize(self, info, logger):
	# 	pass

	def regularize(self, q):
		return util.standard_kl(q).sum()

	def generate(self, N=1):
		q = torch.randn(N, self.latent_dim).to(self.device)
		return self.decode(q)

# class Wasserstein_Autoencoder(Autoencoder):
# 	pass

def grad_penalty(disc, real, fake): # for wgans
	# from "Improved Training of Wasserstein GANs" by Gulrajani et al. (1704.00028)

	B = real.size(0)
	eps = torch.rand(B, *[1 for _ in range(real.ndimension()-1)], device=real.device)

	combo = eps*real.detach() + (1-eps)*fake.detach()
	combo.requires_grad = True
	with torch.enable_grad():
		grad, = autograd.grad(disc(combo).mean(), combo,
		                     create_graph=True, retain_graph=True, only_inputs=True)

	return (grad.contiguous().view(B,-1).norm(2, dim=1) - 1).pow(2).mean()

def grad_penalty_sj(disc, real, fake): # for shannon jensen gans
	# from "Stabilizing Training of GANs through Regularization" by Roth et al. (1705.09367)

	B = real.size(0)

	fake, real = fake.clone().detach(), real.clone().detach()
	fake.requires_grad, real.requires_grad = True, True

	with torch.enable_grad():
		vfake, vreal = disc(fake), disc(real)
		gfake, greal = autograd.grad(vfake.mean() + vreal.mean(),
		                             (fake, real),
		                     create_graph=True, retain_graph=True, only_inputs=True)

	nfake = gfake.view(B,-1).pow(2).sum(-1, keepdim=True)
	nreal = greal.view(B,-1).pow(2).sum(-1, keepdim=True)

	return (vfake.sigmoid().pow(2)*nfake).mean() + ((-vreal).sigmoid().pow(2)*nreal).mean()



def judge(disc, real, fake, out=None, optim=None, disc_gp=None, disc_clip=None):
	if out is None:
		out = util.TensorDict()

	out.verdict_r = disc(real)
	out.verdict_f = disc(fake)
	out.distance = out.verdict_r.mean() - out.verdict_f.mean()

	out.total = -out.distance # disc objective - max distance

	if disc_gp is not None:
		out.disc_gp = grad_penalty(disc, real, fake)
		out.total += disc_gp * out.disc_gp

	if optim is not None:
		optim.zero_grad()
		out.total.backward()
		optim.step()

	# clip discriminator
	if disc_clip is not None:
		for param in disc.parameters():
			param.data.clamp_(-disc_clip, disc_clip)

	return out




