import numpy as np
import torch
import torch.distributions as distrib
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

import omnifig as fig

# from ... import framework as fm
from ...op import framework as fm
from ... import util

from ..features import Prior, Gaussian


@fig.Component('ae')
class Autoencoder(fm.Regularizable, fm.Encodable, fm.Decodable, fm.Model):
	def __init__(self, A, encoder=None, decoder=None, din=None, dout=None, **other):

		if encoder is None:
			encoder = A.pull('encoder')
		if decoder is None:
			decoder = A.pull('decoder')

		reg_wt = A.pull('reg-wt', None)
		reg = A.pull('reg', 'L2' if reg_wt is not None and reg_wt > 0 else None)

		criterion = A.pull('criterion', 'mse')

		viz_latent = A.pull('viz-latent', True)
		viz_rec = A.pull('viz-rec', True)
		viz_enc = A.pull('viz-enc', True)
		viz_dec = A.pull('viz-dec', True)

		super().__init__(A, din=encoder.din, dout=decoder.dout, **other)
		
		self.encoder = encoder
		self.decoder = decoder

		self.criterion = util.get_loss_type(criterion)

		self.reg_wt = reg_wt
		self.reg_fn = util.get_regularization(reg, reduction='sum')

		if self.reg_wt is not None and self.reg_wt > 0:
			self.register_stats('reg-loss')
			self.register_hparam('reg_wt', reg_wt)
		self.register_stats('rec-loss')

		self.latent_dim = self.decoder.din

		self._viz_settings = set()
		if viz_latent:
			self._viz_settings.add('latent')
		if viz_rec:
			self._viz_settings.add('rec')
		if viz_enc:
			self._viz_settings.add('enc')
		if viz_dec:
			self._viz_settings.add('dec')

	def _visualize(self, info, records):
		
		settings = self._viz_settings
		
		if 'enc' in settings and isinstance(self.encoder, fm.Visualizable):
			self.encoder.visualize(info, records)
		if 'dec' in settings and isinstance(self.decoder, fm.Visualizable):
			self.decoder.visualize(info, records)
		
		if 'latent' in info:
			q = info.latent
			if 'latent' in settings and q is not None:
				if isinstance(info.latent, distrib.Distribution):
					q = q.loc

				shape = q.size()
				if len(shape) > 1 and np.product(shape) > 0:
					try:
						records.log('histogram', 'latent-norm', q.norm(p=2, dim=-1))
						records.log('histogram', 'latent-std', q.std(dim=0))
						if isinstance(info.latent, distrib.Distribution):
							records.log('histogram', 'logstd-hist', info.latent.scale.add(1e-5).log().mean(dim=0))
					except ValueError:
						print('\n\n\nWARNING: histogram just failed\n')
						print(q.shape, q.norm(p=2, dim=-1).shape)

		if 'original' in info:
			X = info.original
			if X.ndim == 4 and 'rec' in settings and 'reconstruction' in info:
				B, C, H, W = info.original.shape
				N = min(B, 8)

				R = info.reconstruction
				viz_x, viz_rec = X[:N], R[:N]

				recs = torch.cat([viz_x, viz_rec], 0)
				records.log('images', 'rec', util.image_size_limiter(recs))

	def forward(self, x, ret_q=False):

		q = self.encode(x)
		x = self.decode(q)

		if ret_q:
			return x, q
		return x

	def encode(self, x, **kwargs):
		return self.encoder(x, **kwargs)

	def regularize(self, q, p=None):
		B = q.size(0)
		mag = self.reg_fn(q)
		return mag / B

	def decode(self, q, **kwargs):
		return self.decoder(q, **kwargs)

	def preprocess(self, x):
		return x

	def _rec_step(self, out):

		x = out.original

		B = x.size(0)

		if 'reconstruction' not in out:

			rec, q = self(x, ret_q=True)

			if 'latent' not in out:
				out.latent = q
			out.reconstruction = rec

		rec = out.reconstruction

		loss = self.criterion(rec, x) / B
		out.rec_loss = loss
		self.mete('rec-loss', loss)

		return loss

	def _reg_step(self, out):

		if 'latent' not in out:
			out.latent = self.encode(out.original)

		q = out.latent

		reg_loss = self.regularize(q)
		self.mete('reg-loss', reg_loss)
		out.reg_loss = reg_loss
		return self.reg_wt * reg_loss

	def _process_batch(self, batch, out=None):
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

		x = self.preprocess(x)

		out.original = x
		out.batch = batch

		return out

	def _step(self, batch, out=None):

		out = self._process_batch(batch, out)

		if self.train_me():
			self.optim.zero_grad()

		loss = self._rec_step(out)

		if self.reg_wt is not None and self.reg_wt > 0:
			loss += self._reg_step(out)

		out.loss = loss

		if self.train_me():
			loss.backward()
			self.optim.step()

		return out

class Generative_AE(fm.Generative, Prior, Autoencoder):

	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		self.prior_dim = self.latent_dim

	def generate(self, N=1, prior=None):
		if prior is None:
			prior = self.sample_prior(N)
		return self.decode(prior)


@fig.Component('vae')
class Variational_Autoencoder(Gaussian, Generative_AE):

	def __init__(self, A, **kwargs):

		# mod_check = A.pull('mod_check', True)  # make sure encoder outputs a normal distribution
		# if mod_check:
		# 	mods = A.pull('encoder._mod', None, silent=True)
		# 	if isinstance(mods, (list, tuple, dict)):
		# 		if 'normal' not in mods:
		# 			mods = [*mods, 'normal'] if isinstance(mods, (tuple, list)) else {**mods, 'normal': 10},
		# 			A.push('encoder._mod', mods, silent=True)
		# 	else:
		# 		A.push('encoder._mod.normal', 1)

		A.push('reg', None)  # already taken care of
		wt = A.pull('reg-wt', None, silent=True)
		if wt is None or wt <= 0:
			print('WARNING: VAEs must have a reg_wt (beta), setting to 1')
			A.push('reg-wt', 1)

		super().__init__(A, **kwargs)

	def decode(self, q):
		if isinstance(q, distrib.Distribution):
			q = q.rsample() if self.training else q.mean
		return super().decode(q)

	def regularize(self, q):
		return util.standard_kl(q).sum() / q.loc.size(0)


@fig.Component('wae')
class Wasserstein_Autoencoder(Generative_AE):  # MMD
	def __init__(self, A, **kwargs):
		A.push('reg', None)  # already taken care of
		super().__init__(A, **kwargs)

	def regularize(self, q, p=None):
		if p is None:
			p = self.sample_prior(q.size(0))
		return util.MMD(p, q).sum() / q.size(0)




