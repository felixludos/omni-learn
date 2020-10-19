
import torch

import omnifig as fig

from .. import framework as fm


class Signal:
	def __init__(self, name=None):

		self.name = name

		self.meter = None

	def add_to_stats(self, stats):
		if self.name is None:
			raise Exception(f'signal {self} doesnt have a name for stats')
		stats.new(self.name)
		self.meter = stats

	def step(self):
		val = self._step()

		return val

	def _step(self):
		pass


class RunningNormalization(fm.Model):
	def __init__(self, dim, cmin=-5, cmax=5):
		super().__init__(dim, dim)
		self.dim = dim
		self.n = 0
		self.cmin, self.cmax = cmin, cmax

		self.register_buffer('sum_sq', torch.zeros(dim))
		self.register_buffer('sum', torch.zeros(dim))
		self.register_buffer('mu', torch.zeros(dim))
		self.register_buffer('sigma', torch.ones(dim))

	def update(self, xs):
		xs = xs.view(-1, self.dim)
		self.n += xs.shape[0]
		self.sum += xs.sum(0)
		self.mu = self.sum / self.n

		self.sum_sq += xs.pow(2).sum(0)
		self.mean_sum_sq = self.sum_sq / self.n

		if self.n > 1:
			self.sigma = (self.mean_sum_sq - self.mu**2).sqrt()

	def forward(self, x):
		if self.training:
			self.update(x)
		return ((x - self.mu) / self.sigma).clamp(self.cmin, self.cmax)

