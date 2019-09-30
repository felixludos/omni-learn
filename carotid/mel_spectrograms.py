
import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _tlog10(x):
	return torch.log(x) / torch.log(x.new([10]))

class MEL_SPEC(nn.Module):
	
	def __init__(self, sr=16000, ws=400, hop=None, n_fft=None,
				 pad=0, n_mels=40, window=torch.hann_window, wkwargs={},
				 f_max=None, f_min=0.,
				 stype="power", top_db=-80, scale=True):
		
		super(MEL_SPEC, self).__init__()
		
		# spec
		self.sr = sr
		self.ws = ws
		self.hop = hop if hop is not None else ws // 2
		# number of fft bins. the returned STFT result will have n_fft // 2 + 1
		# number of frequecies due to onesided=True in torch.stft
		self.n_fft = (n_fft-1)*2 if n_fft is not None else ws
		self.n_mels = n_mels
		self.pad = pad
		self.window = window(ws, **wkwargs)
		
		self.norm_fft = False
		
		
		# spec to mel
		self.f_max = f_max if f_max is not None else sr // 2
		self.f_min = f_min
		
		n_fft = self.n_fft // 2 + 1
		
		m_min = 0. if self.f_min == 0 else 2595 * np.log10(1. + (self.f_min / 700))
		m_max = 2595 * np.log10(1. + (self.f_max / 700))
		
		m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
		f_pts = (700 * (10 ** (m_pts / 2595) - 1))
		
		bins = torch.floor(((n_fft - 1) * 2) * f_pts / self.sr).long()
		
		fb = torch.zeros(n_fft, self.n_mels)
		for m in range(1, self.n_mels + 1):
			f_m_minus = bins[m - 1].item()
			f_m = bins[m].item()
			f_m_plus = bins[m + 1].item()
			
			if f_m_minus != f_m:
				fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
			if f_m != f_m_plus:
				fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
		
		self.fb = fb
		
		if self.fb.sum().item() - self.fb.size(-1) != 0:
			print('\n\n***WARNING: MEL spectrogram has some unpopulated rows, please increase fft resolution (n_fft)\n\n')
		
		# power db
		self.norm_spec = None
		self.stype = stype
		self.top_db = -top_db if top_db > 0 else top_db
		self.multiplier = 10. if stype == "power" else 20.
		self.scale = scale
		
	def forward(self, sig):
		
		if self.pad > 0:
			c, n = sig.size()
			new_sig = sig.new_empty(c, n + self.pad * 2)
			new_sig[:, :self.pad].zero_()
			new_sig[:, -self.pad:].zero_()
			new_sig.narrow(1, self.pad, n).copy_(sig)
			sig = new_sig
		
		spec_f = torch.stft(sig, self.n_fft, self.hop, self.ws,
							self.window, center=False,
							normalized=self.norm_fft, onesided=True).transpose(1, 2)
		spec_f /= self.window.pow(2).sum().sqrt()
		spec_f = spec_f.pow(2).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
		
		self.spec_f = spec_f
		#return spec_f
		
		spec = spec_f @ self.fb  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
		
		#return spec_m
		
		self.spec = spec
		
		#print(spec.max())
		
		if self.norm_spec is None:
			spec_db = self.multiplier * _tlog10(spec)
		elif self.norm_spec == 'max':
			spec_db = self.multiplier * _tlog10(spec / spec.max())  # power -> dB
		else:
			spec_db = self.multiplier * _tlog10(spec / self.norm_spec)
			
		if self.top_db is not None:
			spec_db = torch.max(spec_db, spec_db.new([self.top_db]))
		
		spec_db = -spec_db
		
		if self.scale:
			spec_db = spec_db / self.top_db
			spec_db = spec_db + 1.
			
		#print(spec.max(), spec_db.max())
		
		return spec_db
	
	def inverse(self, spec):
		
		if self.scale:
			spec = spec - 1
			spec = spec * self.top_db
			
		assert False

# NOTE: includes forward and inverse transformation - most up to date
class MEL_Spectrogram(torch.nn.Module):
	def __init__(self, ws=1024, hop=512, seq_len=None, n_mels=128, fs=44100, temperature=0, epsilon=1e-10):
		super(MEL_Spectrogram, self).__init__()
		
		assert temperature == 0, 'disabled since soft mel doesnt work for the phase'
		
		self.fs = fs
		self.ws = ws
		self.hop = hop if hop is not None else ws // 2
		self.seq_len = seq_len
		# number of fft bins. the returned STFT result will have n_fft // 2 + 1
		# number of frequecies due to onesided=True in torch.stft
		self.n_fft = (ws // 2) + 1
		self.n_mels = n_mels
		
		m_min = 0.
		m_max = 2595 * np.log10(1. + (self.fs // 2 / 700))
		m_pts = torch.linspace(m_min, m_max, n_mels + 2)
		f_pts = (700 * (10 ** (m_pts / 2595) - 1))
		bins = torch.floor(((self.n_fft - 1) * 2) * f_pts / self.fs).long()
		
		ftr = torch.zeros(self.n_mels, self.n_fft)
		fill = torch.zeros_like(ftr)
		for i, (mn, m, mp) in enumerate(zip(bins, bins[1:], bins[2:])):
			width = mp - mn - 1
			
			if temperature > 0 and width > 1:
				
				center = m.float()  # (mp+mn).float()/2
				
				pos = torch.linspace(mn, mp, width + 2).long()[1:-1]
				
				val = torch.exp(-(pos.float() - center).abs() / temperature)
				
				val = val / val.sum()
				
				ftr[i, pos] = val
			
			# print(mn, m, mp)
			
			# print(val)
			else:
				ftr[i, m] = 1.
			
			fill[i, m:mp] = 1
			
		self.register_buffer('mel_filter', ftr.float())
		self.register_buffer('mel_fill', fill.t().float())
		self.temperature = temperature
		self.epsilon = epsilon
		
		self.forward_transform = None
		scale = self.ws / self.hop
		fourier_basis = np.fft.fft(np.eye(self.ws))
		
		cutoff = int((self.ws / 2 + 1))
		fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
		                           np.imag(fourier_basis[:cutoff, :])])
		forward_basis = torch.tensor(fourier_basis[:, None, :])
		inverse_basis = torch.tensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])
		
		self.register_buffer('forward_basis', forward_basis.float())
		self.register_buffer('inverse_basis', inverse_basis.float())
	
	def stft(self, input_data, ret_phase=False):
		num_batches = input_data.size(0)
		num_samples = input_data.size(-1)
		
		self.num_samples = num_samples
		
		input_data = input_data.view(num_batches, 1, num_samples)
		forward_transform = F.conv1d(input_data,
		                             self.forward_basis,
		                             stride=self.hop,
		                             padding=self.ws)
		cutoff = int((self.ws / 2) + 1)
		real_part = forward_transform[:, :cutoff, :]
		imag_part = forward_transform[:, cutoff:, :]
		
		magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)[...,1:-1]
		
		if ret_phase:
			phase = torch.atan2(imag_part, real_part)[...,1:-1]
			return magnitude, phase
		return magnitude
	
	def _inverse_cossin(self, mag_cossin, lim=None):
		inverse_transform = F.conv_transpose1d(mag_cossin,
		                                       self.inverse_basis,
		                                       stride=self.hop,
		                                       padding=0)
		inverse_transform = inverse_transform[:, :, self.ws:]
		
		if lim is None:
			lim = (self.num_samples if self.seq_len is None else self.seq_len)
		
		inverse_transform = inverse_transform[:, :, :lim]
		return inverse_transform
	
	def stft_inverse(self, magnitude, phase, lim=None): # full spectrum
		return self._inverse_cossin(torch.cat([magnitude * torch.cos(phase),
		                                magnitude * torch.sin(phase)], dim=1), lim=lim)
	
	def mel(self, spec, phase=None):
		
		mel = self.mel_filter @ spec
		mel = mel.add(self.epsilon).log()
		
		if phase is not None:
			phase = self.mel_filter @ phase
			return mel, phase
		
		return mel
	
	def mel_inverse(self, mel, phase):
		imel = mel.exp().sub(self.epsilon)
		mag = self.mel_filter.t() @ imel
		
		phase = self.mel_fill @ phase
		
		return mag, phase
	
	
	def inverse(self, mel, phase, lim=None):
		return self.stft_inverse(*self.mel_inverse(mel, phase), lim=lim)
	
	def forward(self, signal, ret_phase=False):
		spec = self.stft(signal, ret_phase=ret_phase)
		if ret_phase:
			spec, phase = spec
			return self.mel(spec, phase)
		return self.mel(spec)



class STFT(torch.nn.Module):
	def __init__(self, filter_length=1024, hop_length=512):
		super(STFT, self).__init__()

		self.filter_length = filter_length
		self.hop_length = hop_length
		self.forward_transform = None
		scale = self.filter_length / self.hop_length
		fourier_basis = np.fft.fft(np.eye(self.filter_length))

		cutoff = int((self.filter_length / 2 + 1))
		fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
								   np.imag(fourier_basis[:cutoff, :])])
		forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
		inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

		self.register_buffer('forward_basis', forward_basis.float())
		self.register_buffer('inverse_basis', inverse_basis.float())

	def transform(self, input_data):
		num_batches = input_data.size(0)
		num_samples = input_data.size(-1)

		self.num_samples = num_samples

		input_data = input_data.view(num_batches, 1, num_samples)
		forward_transform = F.conv1d(input_data,
									 self.forward_basis,
									 stride = self.hop_length,
									 padding = self.filter_length)
		cutoff = int((self.filter_length / 2) + 1)
		real_part = forward_transform[:, :cutoff, :]
		imag_part = forward_transform[:, cutoff:, :]

		magnitude = torch.sqrt(real_part**2 + imag_part**2)
		phase = torch.atan2(imag_part.data, real_part.data)
		return magnitude, phase

	def _inverse(self, mag_cossin):
		inverse_transform = F.conv_transpose1d(mag_cossin,
		                                       self.inverse_basis,
		                                       stride=self.hop_length,
		                                       padding=0)
		inverse_transform = inverse_transform[:, :, self.filter_length:]
		inverse_transform = inverse_transform[:, :, :self.num_samples]
		return inverse_transform

	def inverse(self, magnitude, phase):
		return self._inverse(torch.cat([magnitude*torch.cos(phase),
											   magnitude*torch.sin(phase)], dim=1))

		

	def forward(self, input_data):
		self.magnitude, self.phase = self.transform(input_data)
		reconstruction = self.inverse(self.magnitude, self.phase)
		return reconstruction

# old (from torchaudio)
#
# class Compose(object):
# 	"""Composes several transforms together.
#
# 	Args:
# 		transforms (list of ``Transform`` objects): list of transforms to compose.
#
# 	Example:
# 		>>> transforms.Compose([
# 		>>>     transforms.Scale(),
# 		>>>     transforms.PadTrim(max_len=16000),
# 		>>> ])
# 	"""
#
# 	def __init__(self, transforms):
# 		self.transforms = transforms
#
# 	def __call__(self, audio):
# 		for t in self.transforms:
# 			audio = t(audio)
# 		return audio
#
# 	def __repr__(self):
# 		format_string = self.__class__.__name__ + '('
# 		for t in self.transforms:
# 			format_string += '\n'
# 			format_string += '    {0}'.format(t)
# 		format_string += '\n)'
# 		return format_string
#
#
#
# class SPECTROGRAM(object):
# 	"""Create a spectrogram from a raw audio signal
#
# 	Args:
# 		sr (int): sample rate of audio signal
# 		ws (int): window size, often called the fft size as well
# 		hop (int, optional): length of hop between STFT windows. default: ws // 2
# 		n_fft (int, optional): number of fft bins. default: ws // 2 + 1
# 		pad (int): two sided padding of signal
# 		window (torch windowing function): default: torch.hann_window
# 		wkwargs (dict, optional): arguments for window function
#
# 	"""
# 	def __init__(self, sr=16000, ws=400, hop=None, n_fft=None,
# 				 pad=0, window=torch.hann_window, wkwargs=None):
# 		if isinstance(window, Variable):
# 			self.window = window
# 		else:
# 			self.window = window(ws) if wkwargs is None else window(ws, **wkwargs)
# 			self.window = Variable(self.window, volatile=True)
# 		self.sr = sr
# 		self.ws = ws
# 		self.hop = hop if hop is not None else ws // 2
# 		# number of fft bins. the returned STFT result will have n_fft // 2 + 1
# 		# number of frequecies due to onesided=True in torch.stft
# 		self.n_fft = (n_fft - 1) * 2 if n_fft is not None else ws
# 		self.pad = pad
# 		self.wkwargs = wkwargs
#
# 	def __call__(self, sig):
# 		"""
# 		Args:
# 			sig (Tensor or Variable): Tensor of audio of size (c, n)
#
# 		Returns:
# 			spec_f (Tensor or Variable): channels x hops x n_fft (c, l, f), where channels
# 				is unchanged, hops is the number of hops, and n_fft is the
# 				number of fourier bins, which should be the window size divided
# 				by 2 plus 1.
#
# 		"""
# 		sig, is_variable = _check_is_variable(sig)
#
# 		assert sig.dim() == 2
#
# 		if self.pad > 0:
# 			c, n = sig.size()
# 			new_sig = sig.new_empty(c, n + self.pad * 2)
# 			new_sig[:, :self.pad].zero_()
# 			new_sig[:, -self.pad:].zero_()
# 			new_sig.narrow(1, self.pad, n).copy_(sig)
# 			sig = new_sig
#
# 		spec_f = torch.stft(sig, self.n_fft, self.hop, self.ws,
# 							self.window, center=False,
# 							normalized=True, onesided=True).transpose(1, 2)
# 		spec_f /= self.window.pow(2).sum().sqrt()
# 		spec_f = spec_f.pow(2).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
# 		return spec_f if is_variable else spec_f.data
#
#
# class F2M(object):
# 	"""This turns a normal STFT into a MEL Frequency STFT, using a conversion
# 	   matrix.  This uses triangular filter banks.
#
# 	Args:
# 		n_mels (int): number of MEL bins
# 		sr (int): sample rate of audio signal
# 		f_max (float, optional): maximum frequency. default: sr // 2
# 		f_min (float): minimum frequency. default: 0
# 	"""
# 	def __init__(self, n_mels=40, sr=16000, f_max=None, f_min=0.):
# 		self.n_mels = n_mels
# 		self.sr = sr
# 		self.f_max = f_max if f_max is not None else sr // 2
# 		self.f_min = f_min
#
# 	def __call__(self, spec_f):
#
# 		spec_f, is_variable = _check_is_variable(spec_f)
# 		n_fft = spec_f.size(2)
#
# 		m_min = 0. if self.f_min == 0 else 2595 * np.log10(1. + (self.f_min / 700))
# 		m_max = 2595 * np.log10(1. + (self.f_max / 700))
#
# 		m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
# 		f_pts = (700 * (10**(m_pts / 2595) - 1))
#
# 		bins = torch.floor(((n_fft - 1) * 2) * f_pts / self.sr).long()
#
# 		fb = torch.zeros(n_fft, self.n_mels)
# 		for m in range(1, self.n_mels + 1):
# 			f_m_minus = bins[m - 1].item()
# 			f_m = bins[m].item()
# 			f_m_plus = bins[m + 1].item()
#
# 			if f_m_minus != f_m:
# 				fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
# 			if f_m != f_m_plus:
# 				fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
#
# 		fb = Variable(fb)
# 		spec_m = torch.matmul(spec_f, fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
# 		return spec_m if is_variable else spec_m.data
#
#
# class SPEC2DB(object):
# 	"""Turns a spectrogram from the power/amplitude scale to the decibel scale.
#
# 	Args:
# 		stype (str): scale of input spectrogram ("power" or "magnitude").  The
# 			power being the elementwise square of the magnitude. default: "power"
# 		top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
# 			is -80.
# 	"""
# 	def __init__(self, stype="power", top_db=None):
# 		self.stype = stype
# 		self.top_db = -top_db if top_db > 0 else top_db
# 		self.multiplier = 10. if stype == "power" else 20.
#
# 	def __call__(self, spec):
#
# 		spec, is_variable = _check_is_variable(spec)
# 		spec_db = self.multiplier * _tlog10(spec / spec.max())  # power -> dB
# 		if self.top_db is not None:
# 			spec_db = torch.max(spec_db, spec_db.new([self.top_db]))
# 		return spec_db if is_variable else spec_db.data
#
#
# class MEL2(object):
# 	"""Create MEL Spectrograms from a raw audio signal using the stft
# 	   function in PyTorch.  Hopefully this solves the speed issue of using
# 	   librosa.
#
# 	Sources:
# 		* https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
# 		* https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
# 		* http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
#
# 	Args:
# 		sr (int): sample rate of audio signal
# 		ws (int): window size, often called the fft size as well
# 		hop (int, optional): length of hop between STFT windows. default: ws // 2
# 		n_fft (int, optional): number of fft bins. default: ws // 2 + 1
# 		pad (int): two sided padding of signal
# 		n_mels (int): number of MEL bins
# 		window (torch windowing function): default: torch.hann_window
# 		wkwargs (dict, optional): arguments for window function
#
# 	Example:
# 		>>> sig, sr = torchaudio.load("test.wav", normalization=True)
# 		>>> sig = transforms.LC2CL()(sig)  # (n, c) -> (c, n)
# 		>>> spec_mel = transforms.MEL2(sr)(sig)  # (c, l, m)
# 	"""
# 	def __init__(self, sr=16000, ws=400, hop=None, n_fft=None,
# 				 pad=0, n_mels=40, window=torch.hann_window, wkwargs=None):
# 		self.window = window(ws) if wkwargs is None else window(ws, **wkwargs)
# 		self.window = Variable(self.window, requires_grad=False)
# 		self.sr = sr
# 		self.ws = ws
# 		self.hop = hop if hop is not None else ws // 2
# 		self.n_fft = n_fft  # number of fourier bins (ws // 2 + 1 by default)
# 		self.pad = pad
# 		self.n_mels = n_mels  # number of mel frequency bins
# 		self.wkwargs = wkwargs
# 		self.top_db = -80.
# 		self.f_max = None
# 		self.f_min = 0.
#
# 	def __call__(self, sig):
# 		"""
# 		Args:
# 			sig (Tensor): Tensor of audio of size (channels [c], samples [n])
#
# 		Returns:
# 			spec_mel_db (Tensor): channels x hops x n_mels (c, l, m), where channels
# 				is unchanged, hops is the number of hops, and n_mels is the
# 				number of mel bins.
#
# 		"""
#
# 		sig, is_variable = _check_is_variable(sig)
#
# 		transforms = Compose([
# 			SPECTROGRAM(self.sr, self.ws, self.hop, self.n_fft,
# 						self.pad, self.window),
# 			F2M(self.n_mels, self.sr, self.f_max, self.f_min),
# 			SPEC2DB("power", self.top_db),
# 		])
#
# 		spec_mel_db = transforms(sig)
#
# 		return spec_mel_db if is_variable else spec_mel_db.data
