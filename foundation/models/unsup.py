
import sys, os, time
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from . import nets
from .. import framework as fm
from .atom import *


class Autoencoder(fm.Regularizable, fm.Encodable, fm.Decodable, fm.Visualizable, fm.Trainable_Model):
	def __init__(self, encoder, decoder, beta=None, criterion=None):

		super().__init__(encoder.din, encoder.dout)

		self.enc = encoder
		self.dec = decoder

		if criterion is None:
			criterion = nn.MSELoss()
		self.criterion = util.get_loss_type(criterion)

		self.beta = beta
		if self.beta is not None:
			self.stats.new('reg')

		self.latent_dim = self.dec.din

	def forward(self, x, ret_q=False):

		q = self.encode(x)
		x = self.decode(q)

		if ret_q:
			return x, q
		return x

	def encode(self, x):
		return self.enc.encode(x)

	def decode(self, q):
		return self.dec.decode(q)

	# def _visualize(self, info, logger):
	# 	if self._viz_counter % 5 == 0:
	# 		# logger.add('histogram', 'latent-norm', info.latent.norm(p=2, dim=-1))
	#
	# 		B, C, H, W = info.original.shape
	# 		N = min(B, 8)
	#
	# 		viz_x, viz_rec = info.original[:N], info.reconstruction[:N]
	#
	# 		recs = torch.cat([viz_x, viz_rec], 0)
	# 		logger.add('images', 'rec', recs)
	#
	# 		# show latent space
	# 		if self.latent_dim >= 2:
	# 			if self.latent_dim > 2:
	# 				x = PCA(n_components=2, copy=False).fit_transform(info.latent.cpu().numpy())
	# 			else:
	# 				x = info.latent.cpu().numpy()
	#
	# 			fig = plt.figure(figsize=(3, 3))
	# 			plt.gca().set_aspect('equal')
	# 			plt.scatter(*x.T, marker='.', s=4, edgecolors='none', alpha=.7)
	# 			# plt.show()
	# 			logger.add('figure', 'latent-space', fig, close=True)
	#
	# 	if 'reg' in info:
	# 		self.stats.update('reg', info.reg)

	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		x, y = batch

		pred, q = self(x, ret_q=True)

		loss = self.criterion(pred, x)

		total = loss

		if self.beta is not None:
			reg = self.regularize(q)
			out.reg = reg
			self.stats.update('reg', reg.detach())
			total += self.beta * reg

		out.total = total

		if self.train_me():
			self.optim_step(total)

		out.reconstruction = pred
		out.original = x
		out.loss = loss
		out.latent = q

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

def grad_penalty(disc, real, fake):
	B = real.size(0)
	eps = torch.rand(B, *[1 for _ in range(real.ndimension()-1)], device=real.device)

	combo = eps*real.detach() + (1-eps)*fake.detach()
	combo.requires_grad = True
	with torch.enable_grad():
		grad, = autograd.grad(disc(combo).mean(), combo,
		                     create_graph=True, retain_graph=True, only_inputs=True)

	return (grad.contiguous().view(B,-1).norm(2, dim=1) - 1).pow(2).mean()

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


class Generative_Adversarial_Network(fm.Regularizable, fm.Generative, fm.Visualizable, fm.Trainable_Model):

	def __init__(self, generator, discriminator, disc_steps=1, disc_clip=None, disc_gp=None):
		super().__init__(generator.din, generator.dout)

		self.gen = generator
		self.disc = discriminator

		self.disc_steps = disc_steps
		assert self.disc_steps >= 1, 'must train discriminator'
		self.disc_clip = disc_clip
		self.disc_gp = disc_gp
		assert self.disc_gp is None or self.disc_clip is None, 'cant regularize with both'

		if self.disc_gp is not None:
			self.stats.new('reg')

		self.step_counter = 0

		self.stats.new('wasserstein')

	def forward(self, N=1):
		return self.generate(N)

	# def _visualize(self, info, logger):
	# 	pass

	def _step(self, batch, out=None):
		self.step_counter += 1

		if out is None:
			out = super()._step(batch, out)

		real = batch[0]
		fake = self.generate(real.size(0))

		out.real = real
		out.fake = fake

		# train discriminator
		out = judge(self.disc, real, fake, out=out,
		            optim=self.optim.disc if self.train_me() else None,
		            disc_gp=self.disc_gp, disc_clip=self.disc_clip)

		self.stats.update('wasserstein', out.distance.detach())
		if 'disc_gp' in out:
			self.stats.update('reg', out.disc_gp.detach())

		out.gen = fake
		out.loss = out.distance # wasserstein distance (neg loss)

		# train generator
		if self.disc_steps is None or self.step_counter % self.disc_steps == 0:
			gen = self.generate(real.size(0))
			pretend = self.disc(gen)

			if self.train_me():
				self.optim.gen.zero_grad()
				(-pretend).mean().backward()
				self.optim.gen.step()

			out.gen = gen
			out.pretend = pretend

		return out


	def generate(self, N=1):
		device = next(iter(self.parameters())).device
		q = torch.randn(N, self.gen.din).to(device)
		return self.gen(q)





