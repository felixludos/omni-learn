
import sys, os, time
from tqdm import tqdm
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import omnifig as fig

from .. import nets
from ... import framework as fm
from ...eval import fid
from ... import viz
from ... import util



@fig.Component('wgan')
class WGAN(fm.Generative, fm.Decodable, fm.Full_Model):
	def __init__(self, config, generator=None, discriminator=None, **other):

		if generator is None:
			generator = config.pull('generator')
		if discriminator is None:
			discriminator = config.pull('discriminator')

		viz_force_gen = config.pull('viz-force-gen', False)
		viz_gen = config.pull('viz-gen', True)
		viz_disc = config.pull('viz-disc', True)
		viz_walks = config.pull('viz-walks', True)
		retain_graph = config.pull('retain-graph', False)
		metric_name = config.pull('metric-name', 'wasserstein')

		fancy_viz_freq = config.pull('fancy_viz_freq', 50)

		if len(other):
			super().__init__(config, **other)
		else:
			super().__init__(generator.din, generator.dout)

		if isinstance(generator, fm.Recordable):
			self.stats.shallow_join(generator.stats)
		if isinstance(discriminator, fm.Recordable):
			self.stats.shallow_join(discriminator.stats)

		self.generator = generator
		self.discriminator = discriminator

		self.latent_dim = self.generator.din
		self.viz_force_gen = viz_force_gen
		self.viz_gen = viz_gen
		self.viz_disc = viz_disc
		self.viz_walks = viz_walks
		self.retain_graph = retain_graph
		self.metric_name = metric_name
		self._fancy_viz_freq = fancy_viz_freq

		self.stats.new(self.metric_name, 'gen-score')

		self.register_attr('total_gen_steps', 0)
		self.register_attr('total_disc_steps', 0)

		self.monitor_fid = config.pull('fid-monitor', '<>fid-dim', None)
		if self.monitor_fid is not None:
			self.stats.new('fid-ref', 'fid-gen')

	def prep(self, datasets):

		inception, fid_stats = None, None
		if 'inception' not in self.volatile and self.monitor_fid is not None:
			dataset = datasets.get('train', None)
			if dataset is None:
				dataset = next(iter(datasets.values()))
			try:
				fid_stats = dataset.get_fid_stats('train', self.monitor_fid)
			except:
				self.monitor_fid = None
				print('WARNING: Failed to load FID stats for this dataset')
			else:
				inception = fid.load_inception_model(dim=self.monitor_fid,
															 device=self.device)
			self.volatile.inception = inception
			self.volatile.fid_stats = fid_stats

		return super().prep(datasets)

	# def _image_size_limiter(self, imgs):
	#     H, W = imgs.shape[-2:]
	#
	#     if H * W < 128*128:
	#         return imgs
	#
	#     return F.interpolate(imgs, size=(128, 128))

	def sample_prior(self, N=1):
		return torch.randn(N, self.latent_dim).to(self.device)

	def decode(self, q):
		return self.generator(q)

	def generate(self, N=1, prior=None):
		if prior is None:
			prior = self.sample_prior(N)
		return self.decode(prior)

	def _evaluate(self, loader, logger=None, A=None, run=None):

		inline = A.pull('inline', False)

		results = {}

		device = A.pull('device', 'cpu')

		# region Default output

		self.stats.reset()
		batches = iter(loader)
		total = 0

		batch = next(batches)
		batch = util.to(batch, device)
		total += batch.size(0)

		with torch.no_grad():
			out = self.test(batch)

		if isinstance(self, fm.Visualizable):
			self._visualize(out, logger)

		results['out'] = out

		for batch in loader:  # complete loader for stats
			batch = util.to(batch, device)
			total += batch.size(0)
			with torch.no_grad():
				self.test(batch)

		results['stats'] = self.stats.export()
		display = self.stats.avgs()  # if smooths else stats.avgs()
		for k, v in display.items():
			logger.add('scalar', k, v)
		results['stats_num'] = total

		# endregion

		# region Prep

		dataset = loader.get_dataset()
		batch_size = loader.get_batch_size()

		n_samples = max(10000, min(len(dataset), 50000))
		n_samples = A.pull('n-samples', n_samples)

		print(f'data: {len(dataset)}, loader: {loader}')

		# endregion

		# region FID

		skip_fid = A.pull('skip-fid', False)

		if not skip_fid:
			fid_dim = A.pull('fid-dim', 2048)

			if self.monitor_fid is not None and 'inception' in self.volatile \
				and fid_dim == self.monitor_fid:
				inception_model = self.volatile.inception
				ds_stats = self.volatile.fid_stats
			else:
				inception_model = fid.load_inception_model(dim=fid_dim, device=device)
				ds_stats = dataset.get_fid_stats('train', fid_dim)

			gen_fn = self.generate
			m, s = fid.compute_inception_stat(gen_fn, inception=inception_model,
													  batch_size=batch_size, n_samples=n_samples,
													  pbar=tqdm if inline else None)
			results['fid_stats'] = [m, s]

			if ds_stats is not None:
				score = fid.compute_frechet_distance(m, s, *ds_stats)
				results['fid'] = score

				logger.add('scalar', 'fid', score)

		else:
			print('Skipping FID evaluation')

		# endregion

		return results

	def _visualize(self, info, logger):

		if self.viz_gen and isinstance(self.generator, fm.Visualizable):
			self.generator.vizualize(info, logger)
		if self.viz_disc and isinstance(self.discriminator, fm.Visualizable):
			self.discriminator.visualize(info, logger)

		if 'latent' in info and info.latent is not None:
			q = info.latent

			shape = q.size()
			if len(shape) > 1 and np.product(shape) > 0:
				try:
					logger.add('histogram', 'latent-norm',
							   q.norm(p=2, dim=-1))
					logger.add('histogram', 'latent-std', q.std(dim=0))
				except ValueError:
					print('\n\n\nWARNING: histogram just failed\n')
					print(q.shape, q.norm(p=2, dim=-1).shape)

		if self.monitor_fid is not None and 'inception' in self.volatile:

			inception = self.volatile.inception
			ref_stats = self.volatile.fid_stats

			real, fake = info.real, info.fake

			B = real.size(0)

			fx = fid.apply_inception(fake, inception).view(B, -1).cpu().numpy()
			mf, sf = np.mean(fx, axis=0), np.cov(fx, rowvar=False)
			fid_ref = fid.compute_frechet_distance(mf, sf, *ref_stats)
			logger.add('scalar', 'fid-ref', fid_ref)

			rx = fid.apply_inception(real, inception).view(B, -1).cpu().numpy()
			mr, sr = np.mean(rx, axis=0), np.cov(rx, rowvar=False)
			fid_gen = fid.compute_frechet_distance(mf, sf, mr, sr)
			logger.add('scalar', 'fid-gen', fid_gen)

		# print(self._viz_counter, self._fancy_viz_freq)

		if 'real' in info:

			B, C, H, W = info.real.shape
			N = min(B, 8)

			if 'real' not in self.volatile and 'real' in info:
				self.volatile.real = info.real
			if 'real' in self.volatile:
				real = self.volatile.real[:N]
				logger.add('images', 'real-img', util.image_size_limiter(real))
				del self.volatile.real

			if 'fake' in self.volatile:
				fake = self.volatile.fake[:N]
				logger.add('images', 'fake-img', util.image_size_limiter(fake))

				if 'gen' not in info:
					info.gen = fake

				del self.volatile.fake

			if 'gen' not in info and self.viz_force_gen:
				with torch.no_grad():
					info.gen = self.generate(N * 2)
			if 'gen' in info:
				viz_gen = info.gen[:2 * N]
				logger.add('images', 'gen-img',
						   util.image_size_limiter(viz_gen))

			# if isinstance(self.generator, fd.Visualizable):
			#     self.generator._visualize(info, logger)
			# if isinstance(self.discriminator, fd.Visualizable):
			#     self.discriminator._visualize(info, logger)

			if self.viz_walks and (not self.training
					or (self._viz_counter>0 and self._viz_counter % self._fancy_viz_freq == 0)):
				# expensive visualizations

				n = 16
				steps = 20
				ntrav = 1

				q = self.sample_prior(n)


				fg, (lax, iax) = plt.subplots(2, figsize=(2 * min(q.size(1) // 20 + 1, 3) + 2, 3))

				viz.viz_latent(q, figax=(fg, lax), )

				Q = q

				D = q.size(-1)

				mn = -2.5 * torch.ones(D, device=q.device)
				mx = 2.5 * torch.ones(D, device=q.device)

				# mn = q.min(-1)[0]
				# mx = q.max(-1)[0]

				vecs = viz.get_traversal_vecs(Q, steps=steps, mnmx=(mn, mx)).contiguous()
				# deltas = torch.diagonal(vecs, dim1=-3, dim2=-1)

				walks = viz.get_traversals(vecs, self.decode, device=self.device).cpu()
				diffs = viz.compute_diffs(walks)

				info.diffs = diffs

				viz.viz_interventions(diffs, figax=(fg, iax))

				# fig.tight_layout()
				border, between = 0.02, 0.01
				plt.subplots_adjust(wspace=between, hspace=between,
									left=5 * border, right=1 - border,
									bottom=border, top=1 - border)

				logger.add('figure', 'distrib', fg)

				full = walks[1:1 + ntrav]
				del walks

				tH, tW = util.calc_tiling(full.size(1), prefer_tall=True)
				B, N, S, C, H, W = full.shape

				# if tH*H > 200: # limit total image size
				# 	pass

				full = full.view(B, tH, tW, S, C, H, W)
				full = full.permute(0, 3, 4, 1, 5, 2, 6).contiguous().view(B, S, C, tH * H, tW * W)

				logger.add('video', 'traversals', full, fps=12)

		logger.flush()


	def _process_batch(self, batch, out=None):

		if out is None:
			out = util.TensorDict()

		out.batch = batch

		if isinstance(batch, torch.Tensor):
			real = batch
		elif isinstance(batch, (tuple, list)):
			real = batch[0]
		elif isinstance(batch, dict):
			real = batch['x']
		else:
			raise NotImplementedError

		out.real = real

		return out

	def _step(self, batch, out=None):

		out = self._process_batch(batch, out)

		if self.train_me():
			self.optim.zero_grad()

		self._disc_step(out)

		if self._take_gen_step():
			if self.train_me():
				self.optim.generator.zero_grad()
			self._gen_step(out)

		del out.batch

		return out

	def _take_gen_step(self):
		return True  # by default always take gen step

	def _verdict_metric(self, vfake, vreal=None):

		if vreal is not None:
			return vreal.mean() - vfake.mean()

		return vfake  # wasserstein metric

	def _disc_loss(self, out):

		vreal = self._verdict_metric(out.vreal)
		vfake = self._verdict_metric(out.vfake)

		diff = self._verdict_metric(vfake, vreal)
		out.loss = diff

		self.stats.update(self.metric_name, diff.detach())

		return -diff  # discriminator should maximize the difference

	def _disc_step(self, out):

		real = out.real

		if 'fake' not in out:
			out.fake = self.generate(real.size(0))
		fake = out.fake

		self.volatile.real = real
		self.volatile.fake = fake

		verdict_real = self.discriminator(real)
		verdict_fake = self.discriminator(fake)

		out.vreal = verdict_real
		out.vfake = verdict_fake

		disc_loss = self._disc_loss(out)
		out.disc_loss = disc_loss

		if self.train_me():
			# self.optim.discriminator.zero_grad()
			disc_loss.backward(retain_graph=self.retain_graph)
			self.optim.discriminator.step()
			self.total_disc_steps += 1

	def _gen_loss(self, out):

		vgen = self._verdict_metric(out.vgen)

		gen_score = vgen.mean()
		out.gen_raw_score = gen_score

		self.stats.update('gen-score', gen_score)

		return -gen_score

	def _gen_step(self, out):

		if 'gen' not in out:
			if 'prior' not in out:
				out.prior = self.sample_prior(out.real.size(0))
			gen = self.generate(prior=out.prior)
			out.gen = gen

		gen = out.gen

		vgen = self.discriminator(gen)
		out.vgen = vgen

		gen_loss = self._gen_loss(out)
		out.gen_loss = gen_loss

		if self.train_me():
			# self.optim.generator.zero_grad()
			gen_loss.backward(retain_graph=self.retain_graph)
			self.optim.generator.step()

			self.total_gen_steps += 1

@fig.Component('gan')
class ShannonJensen_GAN(WGAN):
	def _verdict_metric(self, vfake, vreal=None):
		if vreal is not None:
			return self._verdict_metric(-vfake).mean() + self._verdict_metric(
				vreal).mean()
		return F.sigmoid(vfake).log()  # how real is it?



@fig.AutoModifier('skip-gen')
class SkipGen(WGAN):
	def __init__(self, config, **kwargs):

		disc_steps = config.pull('disc-steps', None)

		if disc_steps is None:
			print('WARNING: not using the skip-gen')

		super().__init__(config, **kwargs)

		self.disc_step_interval = disc_steps

	def _take_gen_step(self):
		return self.disc_step_interval is None \
			   or self.total_disc_steps % self.disc_step_interval == 0


@fig.AutoModifier('info')
class Info(WGAN): # Info-GANs - recover the original samples

	def __init__(self, config):

		super().__init__(config)

		self.rec_wt = config.pull('rec-wt', None)

		if self.rec_wt is not None and self.rec_wt > 0:
			self.rec_criterion = util.get_loss_type(config.pull('rec-criterion', 'mse'))

			if 'rec' in config:
				config.push('rec.din', self.generator.dout)
				config.push('rec.dout', self.generator.din)
			self.rec = config.pull('rec', None)

			self.stats.new('info-loss')

		else:
			self.rec = None


	def _gen_loss(self, out):

		loss = super()._gen_loss(out)

		if self.rec is None:
			return loss

		x = out.gen
		y = out.prior

		if self.train_me():
			self.optim.rec.zero_grad()

		rec = self.rec(x)
		out.rec = rec

		rec_loss = self.rec_criterion(rec, y)

		self.stats.update('info-loss', rec_loss)

		return loss + self.rec_wt * rec_loss

	def _gen_step(self, out):
		super()._gen_step(out)

		if self.train_me():
			self.optim.rec.step()



def grad_penalty(disc, real, fake):  # for wgans
	# from "Improved Training of Wasserstein GANs" by Gulrajani et al. (1704.00028)

	B = real.size(0)
	eps = torch.rand(B, *[1 for _ in range(real.ndimension() - 1)], device=real.device)

	combo = eps * real.detach() + (1 - eps) * fake.detach()
	combo.requires_grad = True
	with torch.enable_grad():
		grad, = autograd.grad(disc(combo).mean(), combo,
							  create_graph=True, retain_graph=True, only_inputs=True)

	return (grad.contiguous().view(B, -1).norm(2, dim=1) - 1).pow(2).mean()


def grad_penalty_sj(disc, real, fake):  # for shannon jensen gans
	# from "Stabilizing Training of GANs through Regularization" by Roth et al. (1705.09367)

	B = real.size(0)

	fake, real = fake.clone().detach(), real.clone().detach()
	fake.requires_grad, real.requires_grad = True, True

	with torch.enable_grad():
		vfake, vreal = disc(fake), disc(real)
		gfake, greal = autograd.grad(vfake.mean() + vreal.mean(),
									 (fake, real),
									 create_graph=True, retain_graph=True, only_inputs=True)

	nfake = gfake.view(B, -1).pow(2).sum(-1, keepdim=True)
	nreal = greal.view(B, -1).pow(2).sum(-1, keepdim=True)

	return (vfake.sigmoid().pow(2) * nfake).mean() + ((-vreal).sigmoid().pow(2) * nreal).mean()


@fig.AutoModifier('grad-penalty')
class GradPenalty(WGAN):
	def __init__(self, config, **kwargs):

		gp_wt = config.pull('gp-wt', None)

		if gp_wt is None:
			print('WARNING: not using the grad-penalty')

		super().__init__(config, **kwargs)

		self.gp_wt = gp_wt

		self.gp_fn = grad_penalty_sj if isinstance(self, ShannonJensen_GAN) else grad_penalty

		self.stats.new('grad-penalty')

	def _grad_penalty(self, out):
		return self.gp_fn(self.discriminator, out.real, out.fake)

	def _disc_loss(self, out):
		loss = super()._disc_loss(out)

		if self.gp_wt is not None and self.gp_wt > 0:
			grad_penalty = self._grad_penalty(out)
			self.stats.update('grad-penalty', grad_penalty)
			loss += self.gp_wt * grad_penalty

		return loss



