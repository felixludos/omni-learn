
import lzma
import numpy as np
import torch

import omnifig as fig

from .. import util
from .framework import FunctionBase


class Compressor:
	def compress(self, data):
		'''tensor -> bytes'''
		pass


	def decompress(self, code):
		'''bytes -> tensor'''
		pass



class LZMACompressor(Compressor):
	def __init__(self, **kwargs):
		super().__init__()
		self._cached_meta = None


	@staticmethod
	def _compress_bytes(bytes):
		return lzma.compress(bytes)


	@staticmethod
	def _decompress_bytes(bytes):
		return lzma.decompress(bytes)


	@staticmethod
	def compress(data, cache_meta=True):
		data = data.detach().cpu().numpy()
		if cache_meta:
			self._cached_meta = data.dtype, data.shape
		return self._compress_bytes(data.tobytes())


	def decompress(self, code, dtype=None, shape=None, cache_stats=True):
		assert (cache_stats and self._cached_meta is not None) or dtype is not None, 'unknown dtype'
		if cache_stats and self._cached_meta is not None:
			if dtype is None:
				dtype = self._cached_meta[0]
			if shape is None:
				shape = self._cached_meta[1]
		if shape is None:
			shape = (-1,)
		return torch.from_numpy(np.frombuffer(self._decompress_bytes(code), dtype=dtype).reshape(*shape))



class Quantizer:
	def quantize(self, x):
		'''tensor -> quantized tensor'''
		return x


	def dequantize(self, z):
		'''quantized tensor -> tensor'''
		return z



class SigfigQuantizer(Quantizer):
	def __init__(self, sigfigs=3, **kwargs):
		super().__init__(**kwargs)
		self.sigfigs = sigfigs


	def quantize(self, x):
		return util.round_sigfigs(x, self.sigfigs)


	def dequantize(self, z):
		return util.sigfig_noise(z, torch.rand_like(z) - 0.5, sigfigs=self.sigfigs)



class QuantizedCompressor(Quantizer, Compressor):
	def compress(self, data):
		return self._compress(self.quantize(data))


	def _compress(self, data, **kwargs):
		return super().compress(data, **kwargs)


	def decompress(self, code, **kwargs):
		return self.dequantize(self._decompress(code, **kwargs))


	def _decompress(self, code, **kwargs):
		return super().decompress(code, **kwargs)



@fig.Component('sigfig-lzma')
class SigfigLZMA(fig.Configurable, QuantizedCompressor, SigfigQuantizer, LZMACompressor):
	def __init__(self, A, sigfigs=None, **kwargs):
		if sigfigs is None:
			sigfigs = A.pull('sigfigs', 3)
		super().__init__(A, sigfigs=sigfigs, **kwargs)


