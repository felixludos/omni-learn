
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
from sklearn.decomposition import PCA

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
	             disc_steps=1, disc_gp=None, criterion=None, feature_match=False, **kwargs):

		_feature_list = []
		_feature_criterion = None
		if feature_match:

			_feature_criterion = util.get_loss_type(criterion)

			def hook(m, input, out):
				_feature_list.append(out)

			for layer in discriminator.conv:
				layer.register_forward_hook(hook)

			criterion = self._feature_match

		super().__init__(encoder, generator, criterion=criterion, **kwargs)

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
			loss += self._feature_criterion(xt, yt)

		self._feature_list.clear()

		return loss

	def _visualize(self, info, logger):
		if self._viz_counter % 5 == 0:
			if 'latent' in info:
				logger.add('histogram', 'latent-norm', info.latent.norm(p=2, dim=-1))
				logger.add('histogram', 'latent-std', info.latent.std(dim=0))

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

		vae_loss *= self.scales['vae']

		if self.beta is not None and self.beta > 0:
			reg = self.regularize(q)
			out.loss_prior = reg
			self.stats.update('reg-vae', reg.detach())
			vae_loss += self.beta * reg

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
		return q.pow(2).sum(-1).mean()

	def sample_prior(self, N=1):
		q = torch.randn(N,self.latent_dim, device=self.device)
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

		splits = q.split(q.size(-1)//self.splits,dim=-1)
		groups = [splits[0]]

		for s in splits[1:]:
			groups.append(s[torch.randperm(q.size(0))])

		hyb = torch.cat(groups, -1)

		return hyb







def get_options():
	parser = train.get_parser()

	parser.add_argument('--disc-steps', type=int, default=1)
	parser.add_argument('--disc-clip', type=float, default=None)
	parser.add_argument('--disc-gp', type=float, default=None)
	parser.add_argument('--fake-gen', action='store_true')
	parser.add_argument('--fake-rec', action='store_true')

	parser.add_argument('--disc-fc', type=int, nargs='+')
	parser.add_argument('--gen-fc', type=int, nargs='+')

	parser.add_argument('--beta', type=float, default=None)
	parser.add_argument('--criterion', type=str, default='bce')

	parser.add_argument('--vae-weight', type=float, default=None)
	parser.add_argument('--enc-gan', action='store_true')
	parser.add_argument('--feature-match', action='store_true')


	parser.add_argument('--vae-scale', type=float, default=1)
	parser.add_argument('--gan-scale', type=float, default=1)


	parser.add_argument('--splits', type=int, default=2)


	return parser

def get_data(args):

	if not hasattr(args, 'dataroot'):
		args.dataroot = '/is/ei/fleeb/workspace/local_data/'

	dataroot = args.dataroot

	if args.dataset == 'dsprites':

		def fmt(batch):
			batch['imgs'] = torch.from_numpy(batch['imgs']).unsqueeze(0)
			return batch['imgs'].float(), batch['latents_values']

		path = os.path.join(dataroot, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
		dataset = datautils.Npy_Loader_Dataset(path, keys=['imgs', 'latents_values'])

		dataset = datautils.Format_Dataset(dataset, fmt)

		args.din = 1, 64, 64

		return dataset,

	elif args.dataset == '3dshapes':

		def fmt(batch):
			batch['images'] = torch.from_numpy(batch['images']).permute(2,0,1)
			return batch['images'].float().div(255), batch['labels']

		path = os.path.join(dataroot, '3dshapes.h5')
		dataset = datautils.H5_Dataset(path, keys=['images', 'labels'])

		dataset = datautils.Format_Dataset(dataset, fmt)

		args.din = 3, 64, 64

		return dataset,


	return train.load_data(args=args)

def get_model(args):

	if not hasattr(args, 'feature_match'): # bwd compatibility
		args.feature_match = False

	model_args = {}

	if args.model_type == 'vae-gan':
		model_cls = VAE_GAN
	elif args.model_type == 'hybrid':
		model_cls = Hybrid_Generator
		model_args['splits'] = args.splits
	else:
		raise Exception('unknown model-type: {}'.format(args.model_type))


	encoder = models.Conv_Encoder(args.din, latent_dim=args.latent_dim,

	                              nonlin=args.nonlin, output_nonlin=None,

	                              channels=args.channels, kernels=args.kernels,
	                              strides=args.strides, factors=args.factors,
	                              down=args.downsampling, norm_type=args.norm_type,

	                              hidden_fc=args.fc)

	generator = models.Conv_Decoder(args.din, latent_dim=args.latent_dim,

	                                nonlin=args.nonlin, output_nonlin='sigmoid',

	                                channels=args.channels[::-1], kernels=args.kernels[::-1],
	                                ups=args.factors[::-1],
	                                upsampling=args.upsampling, norm_type=args.norm_type,

	                                hidden_fc=args.gen_fc)

	discriminator = models.Conv_Encoder(args.din, latent_dim=1,

	                                    nonlin=args.nonlin, output_nonlin=None,

	                                    channels=args.channels, kernels=args.kernels,
	                                    strides=args.strides, factors=args.factors,
	                                    down=args.downsampling, norm_type=args.norm_type,

	                                    hidden_fc=args.disc_fc)

	scales = {
		'vae': args.vae_scale,
		'gan': args.gan_scale,
	}

	model = model_cls(encoder, generator, discriminator, scales=scales, feature_match=args.feature_match,
		             fake_gen=args.fake_gen, fake_rec=args.fake_rec,
		            vae_weight=args.vae_weight, enc_gan=args.enc_gan,
		             disc_steps=args.disc_steps, disc_gp=args.disc_gp, criterion=args.criterion, beta=args.beta, **model_args)

	model.set_optim(util.Complex_Optimizer(
		enc=util.get_optimizer(args.optim_type, encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
		gen=util.get_optimizer(args.optim_type, generator.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
		disc=util.get_optimizer(args.optim_type, discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
	))

	return model




if __name__ == '__main__':

	argv = None

	try:
		train.run_full(get_options, get_data, get_model, argv=argv)

	except KeyboardInterrupt:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()

	except:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()
		ipdb.post_mortem(tb)




