
import sys, os, time, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

import configargparse

import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt
# plt.switch_backend('Agg')
# from sklearn.decomposition import PCA

import foundation as fd
from foundation import models
from foundation import data as datautils
from foundation.models import unsup
from foundation import util
from foundation import train
from foundation import data



class VAE_GAN(fd.Generative, unsup.Autoencoder):

	def __init__(self, encoder, generator, discriminator,
	             fake_gen=False, fake_rec=False, vae_weight=1., enc_gan=False, scales=None,
	             noise_std=None, noisy_gan=False, min_log_std=-3,
	             disc_steps=1, disc_gp=None, criterion=None, feature_match=False, **kwargs):

		_feature_list = []
		_feature_criterion = None
		if feature_match:

			_feature_criterion = util.get_loss_type(criterion, reduction='sum')

			def hook(m, input, out):
				_feature_list.append(out)

			for layer in discriminator.conv:
				layer.register_forward_hook(hook)

			criterion = self._feature_match

		super().__init__(encoder, generator, criterion=criterion, **kwargs)

		self.min_log_std = min_log_std
		self.pred_std = self.enc.dout // 2 == self.dec.din
		self.noise_std = noise_std if not self.pred_std else None
		self.noisy_gan = noisy_gan

		self.disc = discriminator
		self._feature_list = _feature_list
		self._feature_criterion = _feature_criterion

		self.disc_steps = disc_steps
		self.disc_gp = disc_gp if disc_gp is not None else 0
		self.fake_gen = fake_gen # generate new samples when training
		self.fake_rec = fake_rec
		self.fake_hybrid = False
		self.vae_weight = vae_weight
		self.enc_gan = enc_gan
		self.scales = scales


		if self.disc_gp is not None:
			self.stats.new('reg-gan')

		self.stats.new('wasserstein', 'reconstruction')
		self.step_counter = 0

		if self.beta is not None:
			self.stats.remove('reg')
			self.stats.new('reg-vae')

	def _feature_match(self, x, y): # as in VAE-GAN paper

		self._feature_list.clear()
		self.disc(x)
		xf = self._feature_list.copy()

		self._feature_list.clear()
		self.disc(y)
		yf = self._feature_list.copy()

		loss = 0
		for xt, yt in zip(xf, yf):
			loss += self._feature_criterion(xt, yt) / x.size(0)

		self._feature_list.clear()

		return loss

	def _visualize(self, info, logger):
		if self._viz_counter % 5 == 0:
			if 'latent' in info:

				q = info.latent.loc if isinstance(info.latent, distrib.Distribution) else info.latent

				logger.add('histogram', 'latent-norm', q.norm(p=2, dim=-1))
				logger.add('histogram', 'latent-std', q.std(dim=0))

			B, C, H, W = info.original.shape
			N = min(B, 8)

			if 'reconstruction' in info:
				viz_x, viz_rec = info.original[:N], info.reconstruction[:N]

				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', recs)

			# show some generation examples
			if 'gen' in info:
				logger.add('images', 'gen', info.gen[:N*2])
			# try:
			# 	gen = self.generate(2 * N)
			# 	logger.add('images', 'gen', gen)
			# except NotImplementedError:
			# 	pass

			if 'hygen' in info:
				viz_hygen = info.hygen[:2*N]
				logger.add('images', 'hygen', viz_hygen)

			logger.flush()

			# show latent space
			# if self.latent_dim >= 2:
			# 	info.latent = info.latent.sample()
			# 	if self.latent_dim > 2:
			# 		x = PCA(n_components=2, copy=False).fit_transform(info.latent.cpu().numpy())
			# 	else:
			# 		x = info.latent.cpu().numpy()
			#
			# 	fig = plt.figure(figsize=(3, 3))
			# 	plt.gca().set_aspect('equal')
			# 	plt.scatter(*x.T, marker='.', s=6, edgecolors='none', alpha=.7)
			# 	# plt.show()
			# 	logger.add('figure', 'latent-space', fig, close=True)

	def encode(self, x):

		q = super().encode(x)

		if self.noise_std is not None and self.noise_std > 0:
			q = distrib.Normal(loc=q, scale=torch.ones_like(q)*self.noise_std)
		if self.pred_std:
			mu = q.narrow(-1, 0, self.latent_dim)
			logsigma = q.narrow(-1, self.latent_dim, self.latent_dim)
			if self.min_log_std is not None:
				logsigma = logsigma.clamp(min=self.min_log_std)
			sigma = logsigma.exp()

			q = distrib.Normal(loc=mu, scale=sigma)

		return q

	def decode(self, q):
		if isinstance(q, distrib.Distribution):
			q = q.rsample()
		return super().decode(q)

	def _step(self, batch, out=None): # WARNING: not calling super()._step
		self.step_counter += 1
		if out is None:
			out = util.TensorDict()
			# out = super()._step(batch)

		x = batch[0]
		out.original = x

		rec, q = self(x, ret_q=True) # TODO: maybe add noise to encoded latent vector before reconstructing... more VAE, less AE

		out.reconstruction = rec
		out.latent = q

		if self.train_me():
			self.optim.gen.zero_grad()
			self.optim.enc.zero_grad()

		vae_loss = self.criterion(rec, x)
		self.stats.update('reconstruction', vae_loss.detach().clone())

		if self.beta is not None and self.beta > 0:
			reg = self.regularize(q)
			out.loss_prior = reg
			self.stats.update('reg-vae', reg.detach())
			vae_loss += self.beta * reg

		vae_loss *= self.scales['vae']

		if self.scales['vae'] > 0:
			vae_loss.backward(retain_graph=self.fake_rec or self.fake_hybrid)
			if not self.enc_gan:
				self.optim.enc.step()

		# GAN

		real = x
		fakes = self.get_all_fake(rec=rec if self.fake_rec else None,
			                     q=q,
			                     N=real.size(0) if self.fake_gen else None)
		out.gen = fakes[0]

		vreal = self.disc(real)
		vfakes = [self.disc(fake.detach()) for fake in fakes]

		wasserstein = vreal.mean() - sum([vf.mean() for vf in vfakes]) / len(vfakes)
		out.loss_gan = wasserstein
		self.stats.update('wasserstein', wasserstein.detach())

		disc_loss = -wasserstein # max distance
		if self.disc_gp > 0:

			gp_loss = unsup.grad_penalty(self.disc, real, fakes[0])
			out.loss_gp = gp_loss
			self.stats.update('reg-gan', gp_loss.detach())
			disc_loss += self.disc_gp * gp_loss

		if self.train_me():
			self.optim.disc.zero_grad()
			disc_loss.backward()
			self.optim.disc.step()


		if self.scales['gan'] > 0 and (self.disc_steps is None or self.step_counter % self.disc_steps == 0):
			fakes = self.get_all_fake(rec=rec if self.fake_rec else None,
			                          q=q,
			                          N=x.size(0) if self.fake_gen else None, out=out)
			pretend = -sum([self.disc(fake).mean() for fake in fakes]) / len(vfakes)

			if self.train_me():
				(self.scales['gan'] * pretend).backward()

		out.loss = vae_loss/self.scales['vae'] if self.scales['vae'] > 0 else wasserstein

		if self.train_me():
			if self.enc_gan:
				self.optim.enc.step()
			self.optim.gen.step()

		self._feature_list.clear()

		return out

	def get_all_fake(self, rec=None, q=None, N=None, out=None):

		fakes = []

		if N is not None:
			fakes.append(self.generate(N))
		if q is not None:
			try:
				hybrid = self.hybridize(q)
				hygen = self.decode(hybrid)
				if out is not None:
					out.hybrid = hybrid
					out.hygen = hygen
				fakes.append(hygen)
			except NotImplementedError:
				pass
		if rec is not None:
			fakes.append(rec)

		assert len(fakes)

		return fakes


	def hybridize(self, q):
		raise NotImplementedError

	def regularize(self, q): # TODO: maybe try WAE regularization - more like Wasserstein++
		if isinstance(q, distrib.Distribution):
			return util.standard_kl(q).sum().div(q.loc.size(0))
		return q.pow(2).sum().div(q.size(0))

	def sample_prior(self, N=1):
		q = torch.randn(N, self.latent_dim, device=self.device)
		return q

	def generate(self, N=1):
		q = self.sample_prior(N)
		return self.decode(q)

class Hybrid_Generator(VAE_GAN):

	def __init__(self, *args, splits=2, **kwargs):
		super().__init__(*args, **kwargs)
		self.splits = splits
		self.fake_hybrid = True

	def hybridize(self, q):

		if isinstance(q, distrib.Distribution):
			q = q.rsample() if self.noisy_gan else q.loc

		splits = q.split(q.size(-1)//self.splits,dim=-1)
		groups = [splits[0]]

		for s in splits[1:]:
			groups.append(s[torch.randperm(q.size(0))])

		hyb = torch.cat(groups, -1)

		return hyb

class Dropout_Hybrid_Generator(VAE_GAN):

	def __init__(self, *args, prob=.1, prob_max=None, **kwargs):
		super().__init__(*args, **kwargs)

		probs = torch.ones(self.latent_dim)*prob if prob_max is None or prob_max < prob else torch.linspace(prob, prob_max, self.latent_dim)
		self.register_buffer('probs', probs.unsqueeze(0))

		self.fake_hybrid = True

	def hybridize(self, q):
		mu = q.loc if isinstance(q, distrib.Distribution) else q
		sel = (torch.rand_like(mu) - self.probs).gt(0).float()

		if isinstance(q, distrib.Distribution):

			sel_distrib = distrib.Normal(loc=mu*sel, scale=((q.scale-1)*sel+1))
			return sel_distrib.rsample() if self.noisy_gan else sel_distrib.loc

		return mu * sel

class Dropin_Hybrid_Generator(Hybrid_Generator):

	def __init__(self, *args, prob=.1, prob_max=None, **kwargs):
		super().__init__(*args, **kwargs)

		probs = torch.ones(self.latent_dim)*prob if prob_max is None or prob_max < prob else torch.linspace(prob, prob_max, self.latent_dim)

		# print('\n\n', probs, '\n\n')

		self.register_buffer('probs', probs.unsqueeze(0))

		self.fake_hybrid = True

	def hybridize(self, q):

		if isinstance(q, distrib.Distribution):
			q = q.rsample() if self.noisy_gan else q.loc

		hyb = super().hybridize(q)

		sel = (torch.rand_like(q) - self.probs).gt(0).float()

		return q*(1-sel) + hyb*sel

