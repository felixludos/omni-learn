
import numpy as np
import torch
import torch.nn as nn
from itertools import zip_longest
import torch.nn.functional as F
from torch.distributions import Normal
from .. import framework as fm

from .atom import *
from .layers import *

class Double_Encoder(fm.Encodable, fm.Trainable_Model):
	def __init__(self, A):
		
		in_shape = A.pull('in_shape', '<>din')
		# latent_dim = A.pull('latent_dim', '<>dout')
		
		channels = A.pull('channels')
		
		factors = A.pull('factors', 2)
		try:
			len(factors)
		except AttributeError:
			factors = [factors]
		if len(factors) != len(channels):
			factors = factors*len(channels)
		total_factor = np.product(factors)
		
		nonlin = A.pull('nonlin')
		output_nonlin = A.pull('output_nonlin', None)
		
		down_type = A.pull('down_type', 'max')
		norm_type = A.pull('norm_type', None)
		
		
		
		super().__init__()
	
	def encode(self, x):
		return self(x)
	
	def forward(self, x):
		pass

class Double_Decoder(fm.Decodable, fm.Trainable_Model):
	def __init__(self, A):
		super().__init__()
	
	def decode(self, q):
		return self(q)
	
	def forward(self, q):
		pass

class Conv_Encoder(fm.Encodable, fm.Model):

	def __init__(self, in_shape, latent_dim=None, feature_dim=None,
				 nonlin='prelu', output_nonlin=None,
				 channels=[], kernels=3, strides=1, factors=2, down='max',
				 norm_type='instance', output_norm_type=None,
				 hidden_fc=[]):
		
		self.in_shape = in_shape

		cshapes, csets = plan_conv(self.in_shape, channels=channels, kernels=kernels, factors=factors, strides=strides)

		conv_layers = build_conv_layers(csets, factors=factors, pool_type=down, norm_type=norm_type,
										out_norm_type=(output_norm_type if latent_dim is None else norm_type),
										nonlin=nonlin,
										out_nonlin=(output_nonlin if latent_dim is None else nonlin))

		out_shape = cshapes[-1]

		super().__init__(in_shape, out_shape if latent_dim is None else latent_dim)

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

	def encode(self, x):
		return self(x)

	def forward(self, x):
		c = self.conv(x)#.view(-1, self.conv_dim)
		return self.transform_conv_features(c)

class Normal_Conv_Encoder(Conv_Encoder):

	def __init__(self, *args, latent_dim=None, min_log_std=None, **kwargs):

		assert latent_dim is not None, 'must provide a size of the latent space'

		distrib_dim = latent_dim

		super().__init__(*args, latent_dim=2*latent_dim, **kwargs)


		self.distrib_dim = distrib_dim
		self.min_log_std = min_log_std

	def transform_conv_features(self, c):

		q = super().transform_conv_features(c)

		mu, logsigma = q.narrow(-1, 0, self.distrib_dim), q.narrow(-1, self.distrib_dim, self.distrib_dim)

		if self.min_log_std is not None:
			logsigma = logsigma.clamp(min=self.min_log_std)

		return Normal(loc=mu, scale=logsigma.exp())


class Rec_Encoder(Conv_Encoder): # fc before and after recurrence
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
		super().reset()
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
	

class Conv_Decoder(fm.Decodable, fm.Model):

	def __init__(self, out_shape, latent_dim=None, nonlin='prelu', output_nonlin=None,
				 channels=[], kernels=[], ups=[], strides=[], upsampling='deconv', norm_type='instance', output_norm_type=None,
				 hidden_fc=[]):
		
		self.out_shape = out_shape

		dshapes, dsets = plan_deconv(self.out_shape, channels=channels, kernels=kernels, factors=ups,
		                             strides=strides if upsampling == 'deconv' else 1)

		deconv_layers = build_deconv_layers(dsets, sizes=dshapes[1:], nonlin=nonlin, out_nonlin=output_nonlin,
											up_type=upsampling, norm_type=norm_type,
											out_norm_type=output_norm_type)

		super().__init__(dshapes[0] if latent_dim is None else latent_dim, out_shape)

		self.deconv_shape = dshapes[0]
		self.latent_dim = latent_dim if latent_dim is not None else int(np.product(self.deconv_shape))

		self.fc = None
		if latent_dim is not None:
			self.fc = make_MLP(self.latent_dim, int(np.product(self.deconv_shape)), hidden_dims=hidden_fc, nonlin=nonlin, output_nonlin=nonlin)

		self.deconv = nn.Sequential(*deconv_layers)

	def forward(self, q):
		return self.decode(q)

	def decode(self, q):
		if self.fc is not None:
			z = self.fc(q)
		else:
			z = q

		z = z.view(-1, *self.deconv_shape)
		return self.deconv(z)










