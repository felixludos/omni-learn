
import numpy as np
import torch
import torch.nn as nn
from itertools import zip_longest
import torch.nn.functional as F
from .. import framework as fm

from .atom import *
from .layers import *

class Encoder(fm.Model):

	def __init__(self, in_shape, latent_dim=None, feature_dim=None,
				 nonlin='prelu', output_nonlin=None,
				 channels=[], kernels=3, strides=1, factors=2, down='max',
				 norm_type='batch', output_norm_type=None,
				 hidden_fc=[]):
		
		self.in_shape = in_shape

		cshapes, csets = plan_conv(self.in_shape, channels=channels, kernels=kernels, factors=factors, strides=strides)

		conv_layers = build_conv_layers(csets, factors=factors, pool_type=down, norm_type=norm_type,
										out_norm_type=(output_norm_type if latent_dim is None else norm_type),
										nonlin=nonlin,
										out_nonlin=(output_nonlin if latent_dim is None else nonlin))

		out_shape = cshapes[-1]

		super(Encoder, self).__init__(in_shape, out_shape if latent_dim is None else latent_dim)

		self.conv = nn.Sequential(*conv_layers)

		self.conv_shape = out_shape
		self.conv_dim = int(np.product(out_shape))
		if feature_dim is None:
			feature_dim = self.conv_dim
		self.feature_dim = feature_dim
		self.latent_dim = latent_dim if latent_dim is not None else self.feature_dim

		self.fc = None
		if latent_dim is not None:
			self.fc = make_MLP(self.feature_dim, self.latent_dim,
							   hidden_dims=hidden_fc, nonlin=nonlin,
							   output_nonlin=output_nonlin)

	def transform_conv_features(self, c):
		if self.fc is None:
			return c
		c = c.view(-1, self.feature_dim)
		return self.fc(c)

	def forward(self, x):
		c = self.conv(x)#.view(-1, self.conv_dim)

		return self.transform_conv_features(c)

class Rec_Encoder(Encoder): # fc before and after recurrence
	def __init__(self, in_shape, rec_dim,
	             
	             nonlin='prelu', before_fc=[], after_fc=[], out_dim=None,
	             output_nonlin=None, latent_dim=None,
	             
	             rec_type='lstm', rec_layers=1, auto_reset=True,
	             batch_first=False,
	             
	             **kwargs):
		
		super().__init__(in_shape, latent_dim=None, **kwargs)

		
		self.tfm = None
		if latent_dim is None and len(before_fc):
			latent_dim = before_fc[-1]
			before_fc = before_fc[:-1]
		if latent_dim is not None:
			self.tfm = make_MLP(self.feature_dim, latent_dim, hidden_dims=before_fc,
			                       nonlin=nonlin, output_nonlin=nonlin)
		else:
			latent_dim = self.conv_dim
			
		self.latent_dim = latent_dim # input to rec
		
		if out_dim is None:
			if len(after_fc):
				out_dim = after_fc[-1]
				after_fc = after_fc[:-1]
			else:
				out_dim = rec_dim
				rec_dim = None
		if len(after_fc):
			rec_out_dim = rec_dim
			rec_hidden_dim = None
		else:
			rec_out_dim = out_dim
			rec_hidden_dim = rec_dim
			after_fc = None

		self.rec = Recurrence(latent_dim, output_dim=rec_out_dim, hidden_dim=rec_hidden_dim,
		                      auto_reset=auto_reset, batch_first=batch_first,
		                      output_nonlin=(output_nonlin if after_fc is None else None),
		                      rec_type=rec_type, num_layers=rec_layers)
		
		self.dec = None
		if after_fc is not None:
			self.dec = make_MLP(rec_out_dim, out_dim, hidden_dims=after_fc,
			                    nonlin=nonlin, output_nonlin=output_nonlin)
			
		
		self.dout = out_dim

	def reset(self):
		self.rec.reset()
		
	def forward(self, xs):
		if xs.ndimension() == 4:
			xs = xs.unsqueeze(int(self.rec.batch_first))
			
		K, B, C, H, W = xs.size()
		xs = xs.view(K*B,C,H,W)
		
		cs = self.conv(xs)
		
		if self.tfm is not None:
			cs = self.tfm(cs.view(K*B, self.conv_dim))
			
		qs = self.rec(cs.view(K,B,self.latent_dim))
		
		if self.dec is not None:
			qs = self.dec(qs)
			
		return qs
	

class Decoder(fm.Model):

	def __init__(self, out_shape, latent_dim=None, nonlin='prelu', output_nonlin=None,
				 channels=[], kernels=[], ups=[], upsampling='deconv', norm_type='batch', output_norm_type=None,
				 hidden_fc=[]):
		
		self.out_shape = out_shape
		
		deconv, in_shape = make_deconv_net(self.out_shape, nonlin=nonlin, output_nonlin=output_nonlin,
		                                   ups=ups, upsampling=upsampling, norm_type=norm_type,
		                                   out_batch_norm=output_norm_type,
		                                   channels=channels, kernels=kernels)

		super(Decoder, self).__init__(in_shape if latent_dim is None else latent_dim, out_shape)

		self.deconv_shape = in_shape
		self.latent_dim = latent_dim if latent_dim is not None else int(np.product(self.deconv_shape))

		self.fc = None
		if latent_dim is not None:
			self.fc = make_MLP(self.latent_dim, int(np.product(self.deconv_shape)), hidden_dims=hidden_fc, nonlin=nonlin, output_nonlin=nonlin)

		self.deconv = deconv

	def forward(self, q):
		if self.fc is not None:
			z = self.fc(q)
		else:
			z = q

		z = z.view(-1, *self.deconv_shape)
		return self.deconv(z)


class Autoencoder(fm.Unsupervised_Model):
	def __init__(self, shape, latent_dim=None, nonlin='prelu', latent_nonlin=None, recon_nonlin=None,
	             channels=[], kernels=3, factors=1, down='max', up='deconv', norm_type='batch',
	             hidden_fc=[], latent_norm_type=None, output_norm_type=None, criterion=None):
		
		if criterion is None:
			criterion = nn.MSELoss()
		
		super(Autoencoder, self).__init__(criterion, shape, latent_dim)
		
		try:
			len(kernels)
		except TypeError:
			kernels = [kernels] * len(channels)
		
		try:
			len(factors)
		except TypeError:
			factors = [factors] * len(channels)
		
		assert len(channels) == len(kernels)
		assert len(channels) == len(factors)
		
		self.shape = shape
		
		strides = factors if down == 'stride' else [1] * len(channels)
		pools = factors if down == 'max' else False
		
		self.enc = Encoder(self.shape, latent_dim, nonlin=nonlin, output_nonlin=latent_nonlin,
		                   channels=channels, kernels=kernels, strides=strides, pooling=pools, norm_type=norm_type,
		                   hidden_fc=hidden_fc, output_batch_norm=latent_norm_type)
		
		self.latent_dim = self.enc.latent_dim
		
		self.dec = Decoder(self.shape, latent_dim, nonlin=nonlin, output_nonlin=recon_nonlin,
		                   channels=channels[::-1], kernels=kernels[::-1], ups=factors[::-1], upsampling=up,
						   norm_type=norm_type,
		                   hidden_fc=hidden_fc[::-1], output_batch_norm=output_norm_type)
	
	def forward(self, x, ret_q=False):
		
		q = self.encode(x)
		x = self.decode(q)
		
		if ret_q:
			return x, q
		return x
	
	def encode(self, x):
		return self.enc(x)
	
	def decode(self, q):
		return self.dec(q)
	
	def get_loss(self, x, stats=None):
		rec = self(x)
		
		return self.criterion(rec, x)





