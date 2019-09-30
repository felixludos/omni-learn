
import sys, os
import numpy as np
import torch
from itertools import zip_longest
from torch import nn
import torch.nn.functional as F

from .. import framework as fm
from .atom import *
from .layers import *

###################
# Vision models
###################

class Mask_Encoder(fm.Model):
	def __init__(self, shape, num_masks, latent_dim=None,
	             nonlin='prelu', latent_nonlin=None, mask_nonlin='default',
				 channels=[], kernels=3, strides=1, factors=1,
				 down='max', up='deconv', norm_type='instance', feature_dim=None,
				 hidden_fc=[], latent_norm_type=None, out_norm_type=None,
				 skipadd=True,):
		super().__init__(shape, (num_masks,)+shape[1:])

		if mask_nonlin == 'default':
			mask_nonlin = 'sigmoid' if num_masks == 1 else 'softmax2d'

		L = len(channels)

		try:
			assert len(kernels) == L
		except TypeError:
			kernels = [kernels] * L

		try:
			assert len(strides) == L
		except TypeError:
			strides = [strides] * L

		try:
			assert len(factors) == L
		except TypeError:
			factors = [factors] * L

		kernels = [((k, k) if isinstance(k, int) else k) for k in kernels]
		strides = [((s, s) if isinstance(s, int) else s) for s in strides]
		factors = [((f, f) if isinstance(f, int) else f) for f in factors]

		self.skipadd = skipadd

		cshapes, csets = plan_conv(shape, channels=channels, kernels=kernels, factors=factors, strides=strides)

		conv_layers = build_conv_layers(csets, factors=factors, pool_type=down, norm_type=norm_type,
										out_norm_type=(latent_norm_type if latent_dim is not None else norm_type), nonlin=nonlin,
										out_nonlin=(latent_nonlin if latent_dim is not None else nonlin))

		outshape = (num_masks,) + shape[1:]

		dchannels = channels[::-1]
		dkernels = kernels if isinstance(kernels, int) else list(kernels[::-1])
		dfactors = factors if isinstance(factors, int) else list(factors[::-1])
		dstrides = strides if isinstance(strides, int) else list(strides[::-1])

		dfactors = [(f[0]*s[0],f[1]*s[1]) for f,s in zip(dfactors, dstrides)]

		dstrides = [1]*len(dfactors)
		if up == 'deconv':
			dstrides, dfactors = dfactors, dstrides

		dshapes, dsets = plan_deconv(outshape, channels=dchannels, kernels=dkernels,
									 factors=dfactors, strides=dstrides)

		deconv_layers = build_deconv_layers(dsets, sizes=dshapes[1:], up_type=up, norm_type=norm_type,
											out_norm_type=out_norm_type, nonlin=nonlin,
										out_nonlin=mask_nonlin)

		#print(cshapes, dshapes)

		self.conv = nn.ModuleList(conv_layers)

		self.conv_out_dim = int(np.product(cshapes[-1]))
		if feature_dim is None:
			feature_dim = self.conv_out_dim
		self.feature_dim = feature_dim
		self.latent_dim = latent_dim if latent_dim is not None else self.feature_dim
		self.deconv_in_shape = dshapes[0]
		deconv_in_dim = int(np.product(dshapes[0]))

		if latent_dim is not None:
			self.enc_fc = make_MLP(self.feature_dim, self.latent_dim, hidden_dims=hidden_fc,
								   nonlin=nonlin, output_nonlin=latent_nonlin)

			self.dec_fc = make_MLP(self.latent_dim, deconv_in_dim, hidden_dims=hidden_fc[::-1],
								   nonlin=nonlin, output_nonlin=nonlin)

		self.deconv = nn.ModuleList(deconv_layers)

		self.cs = None # intermediate conv outputs

	def encode(self, x):
		# x = x.view(-1, *self.din)

		self.cs = []
		for l in self.conv:
			x = l(x)
			self.cs.append(x)

		# self.deconv_in_shape = x.shape[1:] # update expected deconv in shape based on original shape

		return self.transform_conv_features(x)

	def transform_conv_features(self, c): # after conv
		if hasattr(self, 'enc_fc'):
			return self.enc_fc(c.view(-1, self.feature_dim))
		return c

	def transform_mask_features(self, q): # before deconv
		if hasattr(self, 'dec_fc'):
			q = q.view(-1, self.latent_dim)
			q = self.dec_fc(q)
		return q

	def decode(self, q):
		q = self.transform_mask_features(q)

		m = q.view(-1, *self.deconv_in_shape)

		if self.skipadd:
			assert len(self.cs) == len(self.deconv)
			#ys = self.cs
			ys = self.cs[-2::-1]
		else:
			ys = [None]*len(self.deconv)

		for l, y in zip_longest(self.deconv, ys):
			#print(m.size(), y.size())
			m = l(m, y)

		return m

	def forward(self, x, ret_q=False):

		q = self.encode(x)
		m = self.decode(q)

		if ret_q:
			return m, q

		return m

class Recurrent_Mask_Encoder(Mask_Encoder): # pass conv features directly into a recurrent layer

	def __init__(self, shape, num_masks,
				 rec_dim=128, rec_type='lstm', rec_layers=1,
				 **kwargs):

		super().__init__(shape, num_masks, feature_dim=rec_dim, **kwargs)

		self.rec = Recurrence(self.conv_out_dim, rec_dim, hidden_dim=None,
							  num_layers=rec_layers, rec_type=rec_type,
							  auto_reset=False)

	def reset(self):
		self.rec.reset()

	def transform_conv_features(self, c):
		q = self.rec(c.view(1,-1,self.conv_out_dim)).squeeze(0)
		return super().transform_conv_features(q)


class PoseMask_Encoder(Mask_Encoder):

	def __init__(self, shape, num_masks, pose_dim, pose_fc=[], pose_nonlin=None, feature_dim=None,
	             nonlin='prelu', mask_nonlin=None, conv_mask_features=True,
				 channels=[], batch_norm=True, **kwargs):

		super().__init__(shape, num_masks,
						 nonlin=nonlin, mask_nonlin=mask_nonlin,
						 channels=channels, batch_norm=batch_norm,
						 latent_batch_norm=True, latent_nonlin=nonlin,
						 **kwargs)
		self.dout = pose_dim

		if conv_mask_features:
			self.conv1x1 = ConvLayer(channels[-1], channels[-1], kernel_size=1,
									 batch_norm=batch_norm, nonlin=nonlin)

		self.num_masks = num_masks
		self.pose_dim = pose_dim
		if feature_dim is None:
			feature_dim = self.conv_out_dim
		self.feature_dim = feature_dim

		self.poseencoder = make_MLP(feature_dim, pose_dim, hidden_dims=pose_fc,
										 nonlin=nonlin, output_nonlin=pose_nonlin)

	def transform_mask_features(self, q):
		if hasattr(self, 'conv1x1'):
			q = q.view(-1, *self.deconv_in_shape)
			q = self.conv1x1(q)
		return q

	def transform_to_poses(self, q):
		return self.poseencoder(q.view(-1, self.feature_dim))

	def forward(self, x, ret_masks=False):

		q = self.encode(x)

		p = self.transform_to_poses(q)

		if ret_masks:
			return p, self.decode(q)

		return p






