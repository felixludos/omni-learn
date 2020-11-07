
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

@fig.Component('ae')
class Autoencoder(fm.Regularizable, fm.Encodable, fm.Decodable, fm.Full_Model):
	# def __init__(self, encoder, decoder, reg='L2', reg_wt=0, criterion=None,
	#              viz_latent=True, viz_rec=True):
	def __init__(self, A):

		encoder = A.pull('encoder')
		decoder = A.pull('decoder')

		reg_wt = A.pull('reg_wt', None)
		reg = A.pull('reg', 'L2' if reg_wt is not None and reg_wt > 0 else None)

		criterion = A.pull('criterion', 'mse')

		viz_latent = A.pull('viz-latent', True)
		viz_rec = A.pull('viz-rec', True)

		super().__init__(encoder.din, decoder.dout)

		self.enc = encoder
		self.dec = decoder

		self.criterion = util.get_loss_type(criterion)

		self.reg_wt = reg_wt
		self.reg_fn = util.get_regularization(reg, reduction='sum')

		if self.reg_wt is not None and self.reg_wt > 0:
			self.stats.new('reg-loss')
		self.stats.new('rec-loss')

		self.latent_dim = self.dec.din
		
		self.viz_latent = viz_latent
		self.viz_rec = viz_rec
		
	def _visualize(self, info, logger):
		
		if isinstance(self.enc, fm.Visualizable):
			self.enc.visualize(info, logger)
		if isinstance(self.dec, fm.Visualizable):
			self.dec.visualize(info, logger)

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

		X = info.original
		if X.ndim == 4 and self.viz_rec and 'reconstruction' in info:
			B, C, H, W = info.original.shape
			N = min(B, 8)

			R = info.reconstruction
			viz_x, viz_rec = X[:N], R[:N]

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

	def _rec_step(self, out):

		x = out.original

		B = x.size(0)

		rec, q = self(x, ret_q=True)
		out.latent = q
		out.reconstruction = rec

		loss = self.criterion(rec, x) / B
		out.rec_loss = loss
		self.stats.update('rec-loss', loss)

	def _reg_step(self, out):

		q = out.latent

		reg_loss = self.regularize(q)
		self.stats.update('reg-loss', reg_loss)
		out.reg_loss = reg_loss
		return self.reg_wt * reg_loss

	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		if isinstance(batch, torch.Tensor):
			x = batch
		elif isinstance(batch, (tuple, list)):
			x = batch[0]
		elif isinstance(batch, dict):
			x = batch['x']
		else:
			raise NotImplementedError

		out.batch = batch

		x = self.preprocess(x)
		out.original = x

		loss = self._rec_step(out)

		if self.reg_wt > 0:
			loss += self._reg_step(out)

		out.loss = loss

		if self.train_me():
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out

class Generative_AE(fm.Generative, Autoencoder):

	def sample_prior(self, N=1):
		return torch.randn(N, self.latent_dim).to(self.device)

	def generate(self, N=1, q=None):
		if q is None:
			q = self.sample_prior(N)
		return self.decode(q)


@fig.Component('vae')
class Variational_Autoencoder(Generative_AE):

	def __init__(self, A):

		mod_check = A.pull('mod_check', True) # make sure encoder outputs a normal distribution
		if mod_check:
			mods = A.pull('encoder._mod', None, silent=True)
			if isinstance(mods, (list, tuple, dict)):
				if 'normal' not in mods:
					mods = [*mods, 'normal'] if isinstance(mods, (tuple, list)) else {**mods, 'normal':10},
					A.push('encoder._mod', mods, silent=True)
			else:
				A.push('encoder._mod.normal', 1)

		A.push('reg', None) # already taken care of
		wt = A.pull('reg-wt', None, silent=True)
		if wt is None or wt <= 0:
			print('WARNING: VAEs must have a reg_wt (beta), setting to 1')
			A.push('reg-wt', 1)

		super().__init__(A)

	def decode(self, q):
		if isinstance(q, distrib.Distribution):
			q = q.rsample()
		return super().decode(q)

	def regularize(self, q):
		return util.standard_kl(q).sum()


@fig.Component('wae')
class Wasserstein_Autoencoder(Generative_AE): # MMD
	def __init__(self, A):
		A.push('reg', None)  # already taken care of
		super().__init__(A)

	def regularize(self, q, p=None):
		if p is None:
			p = self.sample_prior(q.size(0))
		return util.MMD(p, q).sum()


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




