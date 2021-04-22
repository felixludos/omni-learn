import sys, os
from pathlib import Path
import subprocess
import pickle
import zipfile
import h5py as hf
import numpy as np
import torch
from omnibelt import unspecified_argument, get_printer, InitWall
import omnifig as fig
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from ... import util
from ...data import register_dataset, Deviced, Batchable, DevicedBase

prt = get_printer(__name__)

class FunctionDataset(DevicedBase, InitWall):
	def __init__(self, function, num, sampler=None, sampler_type='normal',
	             in_memory=False, remove_grads=True, labeled=False, batch_size=128,
	             pbar=None, run_device=None, din=None, dout=None, **kwargs):
		if dout is None:
			dout = function.din if labeled else function.dout
		if din is None:
			din = function.dout
		super().__init__(din=din, dout=dout, **kwargs)
		
		if run_device is None:
			run_device = self.device
		
		if remove_grads:
			for param in function.parameters():
				param.requires_grad = False
		
		self.labeled = labeled
		self.batch_size = batch_size
		self.pbar = pbar
		self.run_device = run_device
		
		self.function = function
		
		if sampler is not None:
			base = sampler.sample(num)
		elif sampler_type == 'normal':
			base = torch.randn(num, function.din)
		elif sampler_type == 'uniform':
			base = torch.rand(num, function.din)
		else:
			raise Exception(f'unknown sampling: {sampler_type}')
		self.register_buffer('base', base)
		
		if in_memory:
			self.register_buffer('samples', self._run_all())
		else:
			self.samples = None
			self.function.to(self.run_device)
		
	def _run_all(self):
		
		loader = DataLoader(TensorDataset(self.base), batch_size=128)
		self.function.to(self.run_device)
		
		if self.pbar is not None:
			loader = self.pbar(loader, desc=f'Generating {len(self.base)} samples')
		
		samples = []
		with torch.no_grad():
			for z, in loader:
				samples.append(self.function(z.to(self.run_device)).to(self.device))
		return torch.cat(samples)

	def to(self, device):
		self.function.to(device)
		super().to(device)

	def __len__(self):
		return len(self.base)

	def __getitem__(self, item):
		
		if self.samples is not None:
			x = self.samples[item]
			if self.labeled:
				return x, self.base[item]
			return x
		
		y = self.base[item]
		with torch.no_grad():
			x = self.function(y.view(-1, y.size(-1)).to(self.run_device)).to(self.device)
		if not isinstance(item, (list, tuple, slice)) and (isinstance(item, int) or not len(item.size())):
			x = x.squeeze(0)
		return (x, y) if self.labeled else x


@register_dataset('function')
class FunctionSamples(Batchable, Deviced, fig.Configurable, FunctionDataset):
	def __init__(self, A, function=None, num=None,
	             sampler=unspecified_argument, sampler_type=unspecified_argument,
	             in_memory=None, remove_grads=None, labeled=None, batch_size=None,
	             pbar=None, run_device=None, **kwargs):

		if sampler is unspecified_argument:
			sampler = A.pull('sampler', None)
		if sampler is None and sampler_type is unspecified_argument:
			sampler_type = A.pull('sampler-type', 'normal')

		if function is None:
			function = A.pull('function')

		if num is None:
			num = A.pull('num-samples', '<>num')
		
		size = function.dout
		if isinstance(size, (list, tuple)):
			size = np.prod(size).item()
		
		if in_memory is None:
			in_memory = A.pull('in-memory', (num*size) < 5e9)
		
		if labeled is None:
			labeled = A.pull('labeled', False)
		
		if batch_size is None:
			batch_size = A.pull('batch-size', 128)
		
		if run_device is None:
			run_device = A.pull('run-device', '<>device')
		
		if pbar is None:
			pbar = A.pull('pbar', None)
		
		if remove_grads is None:
			remove_grads = A.pull('remove-grads', True)
		
		super().__init__(A, function=function, num=num, sampler=sampler, sampler_type=sampler_type,
		                 in_memory=in_memory, remove_grads=remove_grads, labeled=labeled, batch_size=batch_size,
		                 pbar=pbar, run_device=run_device, **kwargs)
