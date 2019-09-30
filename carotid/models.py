
import numpy as np
import torch
import torch.nn as nn

import foundation as fd
import foundation.util as util
from foundation import nets
from mel_spectrograms import MEL_Spectrogram

class Conv_RNN_AutoEncoder(nn.Module):
	
	def __init__(self, in_shape, latent_dim, nonlin='prelu', cls_dim=None,
				 use_batch_norm=True, hop=20, ws=50, channels=256,
				 rec_dim=None, rec_num_layers=1, rec_type='gru', noisy_rec=True):
		super(Conv_RNN_AutoEncoder, self).__init__()
		
		k = ws * 44100 // 1000
		s = hop * 44100 // 1000
		self.conv, out_shape = make_1d_conv_net(in_shape, channels=[channels], kernels=[k], padding=[0],
												strides=[s], batch_norm=[use_batch_norm], nonlin=nonlin,
												output_nonlin=nonlin)
		
		self.conv_out_shape = out_shape
		self.conv_out_dim = np.product(out_shape)
		self.feature_dim = self.conv_out_dim
		
		#print(self.conv_out_shape)
		
		self.enc = nets.RecNet(input_dim=channels, output_dim=latent_dim,
							   rec_type=rec_type,
							   hidden_dim=rec_dim,
							   num_layers=rec_num_layers, batch_first=True)
		
		#assert cls_dim is None or latent_dim > cls_dim, 'latent is not wide enough for classification'
		self.normalized = False
		
		self.dec_input = latent_dim
		if noisy_rec:
			assert (latent_dim - cls_dim) % 2 == 0
			self.dec_input = (latent_dim - cls_dim) // 2 + cls_dim
		
		self.dec = nets.RecNet(input_dim=self.dec_input, output_dim=channels,
							   rec_type=rec_type,
							   hidden_dim=None,
							   num_layers=1, batch_first=True)
		
		self.deconv = nn.Sequential(
			nn.ConvTranspose1d(channels, 128, 170, stride=148, ), # 120, 40 - 256,80  -
			nets.get_nonlinearity(nonlin),
			nn.ConvTranspose1d(128, 1, 24, stride=12, ), # 322 22 - 135,11 -
		)
	
	def forward(self, x, decode=False):
		
		y = self.conv(x)
		
		q = self.enc(y.permute(0, 2, 1), ret_hidden=False)
		
		if decode:
			return q, self.decode(q)
		
		return q
	
	def decode(self, q):
		
		assert q.size(-1) == self.dec_input
		
		z = self.dec(q, ret_hidden=False)
		
		m = self.deconv(z.permute(0, 2, 1))
		
		return m


class Conv_RNN(nn.Module):
	
	def __init__(self, in_shape, out_dim, nonlin='prelu',
				 use_batch_norm=True, hop=20, ws=50, channels=256, rec_in_dim=None,
				 rec_dim=None, rec_num_layers=1, rec_type='gru', use_fc=True):
		super(Conv_RNN, self).__init__()
		
		k = ws * 44100 // 1000
		s = hop * 44100 // 1000
		self.conv, out_shape = make_1d_conv_net(in_shape, channels=[channels], kernels=[k], padding=[0],
												strides=[s], batch_norm=[use_batch_norm], nonlin=nonlin,
												output_nonlin=nonlin)
		
		self.conv_out_shape = out_shape
		self.conv_out_dim = np.product(out_shape)
		self.feature_dim = self.conv_out_dim
		
		if rec_in_dim is None:
			rec_in_dim = channels
		
		self.rec = nets.RecNet(input_dim=rec_in_dim, output_dim=out_dim if use_fc else rec_dim,
							   rec_type=rec_type,
							   hidden_dim=rec_dim if use_fc else None,
							   num_layers=rec_num_layers, batch_first=True)
		
		self.rec_dim = rec_dim
		
		assert use_fc or rec_dim > out_dim, 'rec is not wide enough for output'
		self.use_fc = use_fc
		self.normalized = not use_fc
		self.out_dim = out_dim
	
	def forward(self, x):
		
		y = self.conv(x)
		
		#print(y.size())
		
		if self.rec is not None:
			y = self.rec(y.permute(0, 2, 1), ret_hidden=False)
			
			#print(y.size())
			
			#y = y[:, -1].contiguous()
		
		if not self.use_fc:
			y = y.narrow(-1, 0, self.out_dim)
			y = y + 1
			y = y / y.sum(-1, keepdim=True) + 1e-8
			y = y.log()
		
		return y
	
class ConvRNN_Discriminator(Conv_RNN):
	
	def __init__(self, in_shape, nonlin, cls_dim=None,
				 use_batch_norm=True, hop=20, ws=50, channels=256,
				 rec_dim=128, rec_num_layers=1, rec_type='gru'):
		rin = channels+cls_dim if cls_dim is not None else None
		
		super(ConvRNN_Discriminator, self).__init__(in_shape, out_dim=1, nonlin=nonlin,
				 use_batch_norm=use_batch_norm, hop=hop, ws=ws, channels=channels, rec_in_dim=rin,
				 rec_dim=rec_dim, rec_num_layers=rec_num_layers, rec_type=rec_type, use_fc=True)
		
		self.cls_dim = cls_dim
		
	def forward(self, x, cls=None):
		
		y = self.conv(x).permute(0, 2, 1)
		
		if cls is not None:
			
			B, D = cls.size()
			
			cls = cls.unsqueeze(1).expand(B, y.size(1), D)
			
			y = torch.cat([y, cls], -1)
			
		verdict = self.rec(y)
		
		return verdict


class MEL_Discriminator(nn.Module):
	def __init__(self, mel_dim, cls_dim=None, rec_dim=128, rec_num_layers=1, rec_type='gru'):
		super(MEL_Discriminator, self).__init__()
		
		self.rec_dim = rec_dim
		
		in_dim = 3*mel_dim + cls_dim if cls_dim is not None else 3*mel_dim
		
		self.rec = nets.RecNet(input_dim=in_dim, output_dim=1,
		                       rec_type=rec_type,
		                       hidden_dim=rec_dim,
		                       num_layers=rec_num_layers, batch_first=True)
		
		self.normalized = False
	
	def forward(self, x, cls=None):
		
		if cls is not None:
			
			# print(x.size())
			
			B, D = cls.size()
			
			cls = cls.unsqueeze(1).expand(B, x.size(1), D)
			
			x = torch.cat([x, cls], -1)
			
			# print(x.size(), cls.size())
			# quit()
		
		y = self.rec(x.squeeze(1), ret_hidden=False)
		
		# print(y.size())
		
		# y = y[:, -1].contiguous()
		
		return y

		
class Recurrent_Generator(nn.Module):
	
	def __init__(self, seq_len, latent_dim, cls_dim=None, nonlin='prelu',
	             rec_dim=256, rec_num_layers=2, rec_type='gru',
	             use_batch_norm=False, device='cpu'):
		super(Recurrent_Generator, self).__init__()
		
		self.latent_dim = latent_dim
		self.cls_dim = cls_dim
		self.seq_len = seq_len - 2
		self.device = device
		
		self.gen_input = latent_dim + (cls_dim if cls_dim is not None else 0)
		
		self.inventor = nets.RecNet(input_dim=self.gen_input, output_dim=rec_dim,
		                       rec_type=rec_type,
		                       hidden_dim=None,
		                       num_layers=rec_num_layers, batch_first=True)
		
		in_shape = rec_dim, self.seq_len
		
		channels = [32, 8, 1]
		batch_norm = [use_batch_norm]*len(channels)
		
		kernels = [26, 13, 14]
		strides = [9, 7, 7]
		
		ups = []
		up_type = 'deconv'
		
		padding = [(k - 1) // 2 for k in kernels]
		padding = [0 for _ in kernels]
		
		self.deconv, out_shape = make_1d_deconv_net_rev(in_shape, channels=channels, nonlin=nonlin, batch_norm=batch_norm,
		                                     kernels=kernels, upsampling=ups, output_nonlin='tanh',
                     padding=padding, strides=strides, upsampling_type=up_type)
		
		#print(in_shape, out_shape)
		
		if out_shape[-1] != 132300:
			print('err', 132300 - out_shape[-1])
			quit()
		
	def forward(self, seed=None, num=1, seq_len=None, cls=None, hidden=None):
		
		if seq_len is None:
			seq_len = self.seq_len
			
		B, K = num, seq_len
		
		if cls is not None:
			B = cls.size(0)
		
		if seed is None:
			seed = torch.randn(B, K, self.latent_dim).float().to(self.device)
		
		if cls is not None:
			
			B, D = cls.size()
			cls = cls.unsqueeze(1).expand(B, seed.size(1), D)
			
			seed = torch.cat([seed, cls], -1)
			
		#print(seed.size())
		
		features = self.inventor(seed, hidden=hidden)
		
		#print(features.size())
		
		audio = self.deconv(features.permute(0, 2, 1))
		
		#print(audio.size())
		
		return audio


class MEL_Generator(nn.Module):
	
	def __init__(self, seq_len, latent_dim, gain=5, ws=50, hop=40, fs=44100, n_mels=128, cls_dim=None,
	             rec_dim=256, rec_num_layers=2, rec_type='gru', norm=True, ret_mel=False,
	             use_fc=True, use_conv=False, use_batch_norm=True, device='cpu'):
		super(MEL_Generator, self).__init__()
		
		self.latent_dim = latent_dim
		self.cls_dim = cls_dim
		self.fs = fs
		
		self.seq_len = seq_len * fs // 1000
		self.ws = ws * fs // 1000
		self.hop = hop * fs // 1000
		self.n_mels = n_mels
		self.seed_len = self.seq_len // self.hop - (self.ws // self.hop) + 1
		self.device = device
		self.ret_mel = ret_mel
		
		self.use_fc = use_fc and not use_conv
		
		self.gain = gain
		self.norm = norm
		print('Norm:',self.norm)
		
		
		self.gen_input = latent_dim + (cls_dim if cls_dim is not None else 0)
		
		assert self.use_fc or use_conv or 3*n_mels <= rec_dim, 'not enough rec dim'
		
		self.inventor = nets.RecNet(input_dim=self.gen_input, output_dim=3*n_mels if self.use_fc else rec_dim,
		                            rec_type=rec_type, output_nonlin='tanh',
		                            hidden_dim=rec_dim if self.use_fc else None,
		                            num_layers=rec_num_layers, batch_first=True)
		
		
		self.conv = None
		# if use_conv:
		# 	assert False, 'not ready yet'
		# 	assert not use_batch_norm
		# 	self.conv = nn.Sequential(
		# 		nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
		# 		nn.Tanh()
		# 	)
		
		
		self.spec = MEL_Spectrogram(ws=self.ws, hop=self.hop, fs=self.fs, n_mels=n_mels)
		
		
	def forward(self, seed=None, num=1, seq_len=None, cls=None, hidden=None):
		
		if seq_len is None:
			seed_len = self.seed_len
			seq_len = self.seq_len
		else:
			seq_len = seq_len * self.fs // 1000
			seed_len = seq_len // self.hop - (self.ws // self.hop) + 1
		
		B, K = num, seed_len+(0 if self.ret_mel else 1) # gen a slightly longer sequence and then crop (in spec)
		
		if cls is not None:
			B = cls.size(0)
		
		if seed is None:
			seed = torch.randn(B, K, self.latent_dim).float().to(self.device)
		
		if cls is not None:
			B, D = cls.size()
			cls = cls.unsqueeze(1).expand(B, seed.size(1), D)
			
			seed = torch.cat([seed, cls], -1)
		
		#print(seed.size())
		
		features = self.inventor(seed, hidden=hidden)
		
		#print(features.size())
		
		mel, cos, sin = features.narrow(-1, 0, self.n_mels), features.narrow(-1, self.n_mels, self.n_mels), features.narrow(-1, 2*self.n_mels, self.n_mels),
		
		self.mel = mel * self.gain
		self.phase = torch.atan2(sin, cos)
		
		#print(mel.abs().max(), sin.abs().max(), cos.abs().max())
		#print(self.mel.size(), self.phase.size()# )
		
		if self.ret_mel:
			return self.mel, self.phase
		
		audio = self.spec.inverse(self.mel.permute(0, 2, 1), self.phase.permute(0, 2, 1), lim=seq_len)#.clamp(-1,1)
		
		#print(self.mel.abs().max().item(), audio.abs().max().item())
		
		mx = audio.abs().max(-1, keepdim=True)[0]
		
		if self.norm and mx.sum()>0:
			audio = 0.95 * audio / mx
			#print('***normed')
		
		#print(audio.size())
		
		return audio
	
	
class MEL_RNN(nn.Module):
	def __init__(self, in_shape, out_dim, rec_dim=128, rec_num_layers=1, rec_type='gru', use_fc=True):
		super(MEL_RNN, self).__init__()
		
		self.rec_dim = rec_dim
		
		self.rec = nets.RecNet(input_dim=in_shape[-1], output_dim=out_dim if use_fc else rec_dim,
							   rec_type=rec_type,
							   hidden_dim=rec_dim if use_fc else None,
							   num_layers=rec_num_layers, batch_first=True)
		
		
		
		assert use_fc or self.rec_dim > out_dim, 'rec is not wide enough for output'
		self.use_fc = use_fc
		self.normalized = not use_fc
		self.out_dim = out_dim
		
	def forward(self, x):
		
		y = self.rec(x.squeeze(1), ret_hidden=False)
		
		# print(y.size())
		
		# y = y[:, -1].contiguous()
		
		if not self.use_fc:
			y = y.narrow(-1, 0, self.out_dim)
			y = y + 1
			y = y / y.sum(-1, keepdim=True) + 1e-8
			y = y.log()
		
		return y

class MEL_Encoder(nn.Module):
	
	def __init__(self, in_shape, out_dim, nonlin='prelu',
	             use_batch_norm=True,
	             fc_dims=[], ):
		super(MEL_Encoder, self).__init__()
		
		channels = [16, 16, 32, 32, 64]
		self.conv, self.conv_out_shape = nets.make_conv_net(in_shape,
				                               channels=channels,
				                               kernels=[5]+[3]*(len(channels)-1),
				                               pooling=[True]*len(channels),
											   batch_norm=[use_batch_norm]*len(channels),
		                                       nonlin=nonlin, output_nonlin=nonlin)
		
		self.pool = nn.MaxPool2d(2)
		
		self.conv_out_shape = (channels[-1], 4, 2)
		print(in_shape, self.conv_out_shape)
		self.feature_dim = np.product(self.conv_out_shape)
		
		self.fc = nets.make_MLP(input_dim=self.feature_dim, output_dim=out_dim,
								hidden_dims=fc_dims, nonlin=nonlin)
		
		self.normalized = False
		
	def forward(self, x):
		
		y = self.pool(self.conv(x))
		
		# print(x.size(), y.size())
		# quit()
		
		try:
		
			y = y.view(-1, self.feature_dim)
		except Exception as e:
			print(y.size(), self.conv_out_shape, self.feature_dim)
			raise e
		
		y = self.fc(y)
		
		#print(y.size())
		#quit()
		
		return y


def make_1d_conv_net(in_shape, channels=[], output_nonlin=None, nonlin='prelu', batch_norm=[], kernels=[], pooling=[],
                  padding=[], strides=[], dilation=[], ret_output_shape=True):
	assert len(channels) > 0
	assert len(kernels) == len(channels)
	assert len(pooling) == 0 or len(pooling) == len(channels)
	assert len(padding) == 0 or len(padding) == len(channels)
	assert len(strides) == 0 or len(strides) == len(channels)
	assert len(batch_norm) == 0 or len(batch_norm) == len(channels)
	assert len(dilation) == 0 or len(dilation) == len(channels)
	
	if len(pooling) != len(channels):
		pooling = [False] * len(channels)
	if len(padding) != len(channels):
		padding = [(k - 1) // 2 for k in kernels]
	if len(strides) != len(channels):
		strides = [1] * len(channels)
	if len(batch_norm) != len(channels):
		batch_norm = [True] * len(channels)
	if len(dilation) != len(channels):
		dilation = [1] * len(channels)
	nonlins = [nonlin] * len(channels)
	nonlins[-1] = output_nonlin
	
	layers = []
	
	channels = [in_shape[0]] + channels
	shape = in_shape
	
	for in_c, out_c, pool, kernel, pad, stride, nonlin, bn, dil in zip(channels, channels[1:], pooling, kernels, padding,
	                                                              strides, nonlins, batch_norm, dilation):
		layers.append(nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, dilation=dil))
		
		C, L = shape
		shape = out_c, (L + 2*pad - dil * (kernel-1) - 1) // stride + 1
		
		if pool:
			layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
			shape = shape[0], (shape[1] + 1) // 2, (shape[2] + 1) // 2
		if bn:
			layers.append(nn.BatchNorm1d(out_c))
		if nonlin is not None:
			layers.append(nets.get_nonlinearity(nonlin))
	
	net = nn.Sequential(*layers)
	
	if ret_output_shape:
		return net, shape
	
	return net


def make_1d_deconv_net(out_shape, channels=[], output_nonlin=None, nonlin='prelu', batch_norm=[], kernels=[], upsampling=[],
                     padding=[], strides=[], upsampling_type='deconv', ret_output_shape=True):
	assert len(channels) > 0
	assert len(kernels) == len(channels)
	assert len(upsampling) == 0 or len(upsampling) == len(channels)
	assert len(padding) == 0 or len(padding) == len(channels)
	assert len(strides) == 0 or len(strides) == len(channels)
	assert len(batch_norm) == 0 or len(batch_norm) == len(channels)
	assert upsampling_type in {'deconv', 'linear', 'nearest'}
	
	if len(upsampling) != len(channels):
		upsampling = [False] * len(channels)
	if len(padding) != len(channels):
		padding = [(k - 1) // 2 for k in kernels]
	if len(strides) != len(channels):
		strides = [1] * len(channels)
	if len(batch_norm) != len(channels):
		batch_norm = [True] * len(channels)
	nonlins = [nonlin] * len(channels)
	nonlins[-1] = output_nonlin
	
	layers = []
	
	channels = channels + [out_shape[0]]
	shape = out_shape
	
	for in_c, out_c, kernel, pad, stride, nonlin, bn, up in reversed(
			list(zip(channels, channels[1:], kernels, padding, strides, nonlins, batch_norm, upsampling))):
		
		if nonlin is not None:
			layers.append(nets.get_nonlinearity(nonlin))
			
		if bn:
			layers.append(nn.BatchNorm1d(out_c))
			
		C, L = shape
		
		if upsampling_type == 'linear' or upsampling_type == 'nearest':
			
			layers.append(nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad))
			
			layers.append(nn.Upsample(size=up, mode=upsampling_type))
			C, L = in_c, (up-1) * stride - 2 * pad + kernel
		
		if upsampling_type == 'deconv':
			layers.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad))
			
			C, L = in_c, (L - 1) * stride - 2 * pad + kernel
		
		shape = C, L
		
		
	net = nn.Sequential(*layers[::-1])
	
	if ret_output_shape:
		return net, shape
	
	return net


def make_1d_deconv_net_rev(in_shape, channels=[], output_nonlin=None, nonlin='prelu', batch_norm=[], kernels=[],
                       upsampling=[],
                       padding=[], strides=[], upsampling_type='deconv', ret_output_shape=True):
	assert len(channels) > 0
	assert len(kernels) == len(channels)
	assert len(upsampling) == 0 or len(upsampling) == len(channels)
	assert len(padding) == 0 or len(padding) == len(channels)
	assert len(strides) == 0 or len(strides) == len(channels)
	assert len(batch_norm) == 0 or len(batch_norm) == len(channels)
	assert upsampling_type in {'deconv', 'linear', 'nearest'}
	
	if len(upsampling) != len(channels):
		upsampling = [False] * len(channels)
	if len(padding) != len(channels):
		padding = [(k - 1) // 2 for k in kernels]
	if len(strides) != len(channels):
		strides = [1] * len(channels)
	if len(batch_norm) != len(channels):
		batch_norm = [True] * len(channels)
	nonlins = [nonlin] * len(channels)
	nonlins[-1] = output_nonlin
	
	layers = []
	
	channels = [in_shape[0]] + channels
	shape = in_shape
	
	for in_c, out_c, kernel, pad, stride, nonlin, bn, up in \
			zip(channels, channels[1:], kernels, padding, strides, nonlins, batch_norm, upsampling):
		
		C, L = shape
		
		if upsampling_type == 'linear' or upsampling_type == 'nearest':
			
			if up:
				layers.append(nn.Upsample(size=up, mode=upsampling_type))
				C, L = C, up
			
			layers.append(nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad))
			C, L = out_c, (L + 2*pad - kernel) // stride + 1
		
		if upsampling_type == 'deconv':
			layers.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad))
			
			C, L = out_c, (L - 1) * stride - 2 * pad + kernel
		
		shape = C, L
		
		if bn:
			layers.append(nn.BatchNorm1d(out_c))
			
		if nonlin is not None:
			layers.append(nets.get_nonlinearity(nonlin))
	
	net = nn.Sequential(*layers)
	
	if ret_output_shape:
		return net, shape
	
	return net
	
class Wav_Encoder(nn.Module):
	
	def __init__(self, in_shape, out_dim, nonlin='prelu',
	             use_batch_norm=True,
	             rec_dim=None, rec_num_layers=1, rec_type='gru',
	             fc_dims=None, ):
		super(Wav_Encoder, self).__init__()
		
		channels = [32, 64, 128]
		D = len(channels)
		k = 40
		s = k // 2
		self.conv, out_shape = make_1d_conv_net(in_shape, channels=channels, kernels=[k]*D, padding=[0]*D,
		                             strides=[s]*D, batch_norm=[use_batch_norm]*D, nonlin=nonlin, output_nonlin=nonlin)
		
		self.conv_out_shape = out_shape
		self.conv_out_dim = np.product(out_shape)
		self.feature_dim = self.conv_out_dim
		
		self.rec = None
		if rec_dim is not None:
			self.rec = nets.RecNet(input_dim=out_shape[0], output_dim=rec_dim, rec_type=rec_type,
								   output_nonlin=None if fc_dims is None else nonlin,
								   num_layers=rec_num_layers, batch_first=True)
			self.feature_dim = rec_dim
		
		assert fc_dims is not None or rec_dim > out_dim, 'rec is not wide enough for output'
		
		self.fc = None
		if fc_dims is not None:
			self.fc = nets.make_MLP(input_dim=self.feature_dim, output_dim=out_dim, hidden_dims=fc_dims,
									nonlin=nonlin)
			
		self.normalized = self.fc is not None
		self.out_dim = out_dim
		
	def forward(self, x):
		
		y = self.conv(x)
		
		if self.rec is not None:
			y = self.rec(y.permute(0,2,1))
			y = y[:, -1].contiguous()
			
		if self.fc is not None:
			y = self.fc(y.view(-1, self.feature_dim))
		else:
			y = y.narrow(-1, 0, self.out_dim)
			y += 1
			y /= y.sum(-1, keepdim=True) + 1e-8
			y = y.log()

		return y