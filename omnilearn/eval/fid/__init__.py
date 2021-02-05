
from .fid import load_inception_model, compute_frechet_distance, compute_inception_stat, apply_inception
from . import compute

import omnifig as fig
from ... import util

@fig.Component('fid')
class ComputeFID(util.Deviced):  # TODO: turn into an alert and stats client
	def __init__(self, A, dim=None, ret_stats=None,
	             batch_size=None, n_samples=None, **kwargs):
		
		if dim is None:
			dim = A.pull('dim', 2048)
		
		skip_load = A.pull('skip_load', False)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 50)
		
		if n_samples is None:
			n_samples = A.pull('n_samples', 50000)
		
		if ret_stats is None:
			ret_stats = A.pull('ret_stats', True)
		
		pbar = A.pull('pbar', None)  # TODO: add progress bar for fid computation
		
		super().__init__(A, **kwargs)
		
		self.batch_size = batch_size
		self.n_samples = n_samples
		self.ret_stats = ret_stats
		self.dim = dim
		
		self.pbar = pbar
		
		self.inception = None
		if not skip_load:
			self._load_inception()
		
		self.baseline_stats = None
	
	def _load_inception(self, force=False):
		if self.inception is None or force:
			print(f'Loading inception model dim={self.dim}')
			self.inception = load_inception_model(self.dim, self.get_device())
	
	def set_baseline_stats(self, stats=None):
		self.baseline_stats = stats
	
	def compute_stats(self, generate, batch_size=None, n_samples=None, name=None, pbar=None):
		
		self._load_inception()
		
		if batch_size is None:
			batch_size = self.batch_size
		if n_samples is None:
			n_samples = self.n_samples
		if pbar is None:
			pbar = self.pbar
		
		stats = compute_inception_stat(generate, inception=self.inception, pbar=pbar,
		                               batch_size=batch_size, n_samples=n_samples, name=name)
		
		return stats
	
	def compute_distance(self, stats1, stats2=None):
		
		if stats2 is None:
			stats2 = self.baseline_stats
		
		assert stats2 is not None, 'no base stats found'
		return compute_frechet_distance(*stats1, *stats2)
