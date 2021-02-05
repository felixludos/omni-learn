import math
import torch

import omnifig as fig

from ..op import framework as fm
from ..op.clock import Freq, Reg, Savable

from .. import util


class Scheduler(util.Value, Savable, Reg, Freq):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		self.start = A.pull('start', 0)
		self.stop = A.pull('stop', None)
		
		self.end = A.pull('end', '<>end', None)

		self.initial = self.get()
	
	def extra_repr(self):
		return {'end': self.end}
	
	def __repr__(self):
		extra = [f'    {k}: {v},' for k, v in self.extra_repr().items() if v is not None]
		
		return '{}[{}:{}:{}]({})'.format(self.__class__.__name__, '' if self.start is None else self.start,
		                                 '' if self.stop is None else self.stop, '' if self.freq is None else self.freq,
		                                 ('\n    ' + '\n'.join(extra) + '\n') if len(extra) else '')
	
	def __str__(self):
		return repr(self)
	
	def check(self, tick, info=None):
		return super().check(tick, info=info) \
		       and (self.start is None or tick >= self.start) \
		       and (self.stop is None or tick < self.stop)
	
	def activate(self, tick, info=None):
		x = self.step(self.get(), self.initial, tick, info=info)
		self.set(x)
		return x
	
	def state_dict(self):
		return {'end': self.end, 'start': self.start, 'stop': self.stop, 'val':self.get()}
	
	def load_state_dict(self, data):
		self.set(data['val'])
		if data is not None:
			for k, v in data.items():
				if k != 'val':
					setattr(self, k, v)
	
	def step(self, x, x0, tick, info=None):
		raise NotImplementedError


@fig.Component('scheduler/step')
class StepScheduler(Scheduler):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		self.factor = A.pull('factor', '<>gamma')
	
	def state_dict(self):
		data = super().state_dict()
		data['factor'] = self.factor
		return data
	
	def step(self, x, x0, tick, info=None):
		return self.factor * x


@fig.Component('scheduler/multi-step')
class MultiStepScheduler(StepScheduler):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		self.freq = None
		
		num = A.pull('num-steps', 1)
		limit = self.stop
		if limit is None:
			limit = A.pull('limit')
		
		size = limit // num
		self.steps = set([size * step for step in range(1, num)])
	
	def state_dict(self):
		data = super().state_dict()
		data['steps'] = self.steps
		return data
	
	def check(self, tick, info=None):
		return super().check(tick, info=info) and tick in self.steps


class FunctionScheduler(Scheduler):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		assert self.end is not None, 'must have a min for exponential scheduler'
		
		self.limit = A.pull('limit', '<>stop')
		
		self.skew = math.exp(A.pull('skew', 0)) # +/- -> left/right
		self.kurtosis = 3. ** A.pull('kurtosis', 0) # +/- -> squeeze/stretch middle
		self.constrain_progress = A.pull('constrain-progress', False)
	
	def state_dict(self):
		data = super().state_dict()
		data.update({'limit': self.limit, 'skew': self.skew, 'kurtosis': self.kurtosis,
		             'constrain_progress': self.constrain_progress})
		return data
	
	def step(self, x, x0, tick, info=None):
		progress = (tick / self.limit) #** self.skew
		progress = 2*progress - 1
		progress = progress**self.kurtosis if progress >= 0 else -(-progress)**self.kurtosis
		progress = 0.5*progress + 0.5
		progress **= self.skew
		if self.constrain_progress:
			progress = max(0., min(progress, 1.))
		val = self._func(progress, x0)
		return val
	
	def _func(self, progress, x0):
		raise NotImplementedError


@fig.Component('scheduler/cos')
class CosScheduler(FunctionScheduler):
	def __init__(self, A, **kwargs):
		trig = A.pull('trig', 'cos')
		nodes = A.pull('half-periods', 1)
		super().__init__(A, **kwargs)
		if trig == 'cos':
			self.trig = math.cos
		elif trig == 'sin':
			self.trig = math.sin
		else:
			raise ValueError(trig)
		self.nodes = nodes
		
	def _func(self, progress, x0):
		return (x0 - self.end) * (self.trig(self.nodes * math.pi * progress) + 1) / 2 + self.end


@fig.Component('scheduler/exp')
class ExpScheduler(FunctionScheduler):
	def _func(self, progress, x0):
		return x0 * (self.end / x0) ** progress


@fig.Component('scheduler/lin')
class LinScheduler(FunctionScheduler):
	def _func(self, progress, x0):
		return x0 - (x0 - self.end) * progress


#### old



# class Scheduler(Freq):
# 	def __init__(self, A, **kwargs):
# 		super().__init__(A, **kwargs)
# 		self.start = A.pull('start', 0)
# 		self.stop = A.pull('stop', None)
#
# 		self.min_lr = A.pull('min_lr', None)
#
# 		self.optimizer = None
#
# 	def extra_repr(self):
# 		return {'min': self.min_lr}
#
# 	def __repr__(self):
# 		extra = [f'    {k}: {v},' for k,v in self.extra_repr().items() if v is not None]
#
# 		return '{}[{}:{}:{}]({})'.format(self.__class__.__name__, '' if self.start is None else self.start,
# 		                             '' if self.stop is None else self.stop, '' if self.freq is None else self.freq,
# 		                                 ('\n    '+'\n'.join(extra)+'\n') if len(extra) else '')
#
# 	def prep(self, optim):
# 		self.optimizer = optim
# 		self._initial_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
#
# 	def check(self, tick, info=None):
# 		return super().check(tick, info=info) \
# 		        and (self.start is None or tick >= self.start) \
# 		        and (self.stop is None or tick < self.stop)
#
# 	def activate(self, tick, info=None):
# 		lr = None
# 		for param_group, lr0 in zip(self.optimizer.param_groups, self._initial_lrs):
# 			lr = param_group['lr']
# 			lr = self.compute_lr(lr, lr0, tick, info=info)
# 			if self.min_lr is not None:
# 				lr = max(self.min_lr, lr)
# 			param_group['lr'] = lr
#
# 		return lr
#
# 	def state_dict(self):
# 		return {'min_lr':self.min_lr, 'start':self.start, 'stop':self.stop}
#
# 	def load_state_dict(self, data):
# 		if data is not None:
# 			for k, v in data.items():
# 				setattr(self, k, v)
#
# 	def compute_lr(self, lr, lr0, tick, info=None):
# 		raise NotImplementedError
#
#
# @fig.Component('scheduler/step')
# class StepScheduler(Scheduler):
# 	def __init__(self, A, **kwargs):
# 		super().__init__(A, **kwargs)
#
# 		self.factor = A.pull('factor', '<>gamma')
#
# 	def state_dict(self):
# 		data = super().state_dict()
# 		data['factor'] = self.factor
# 		return data
#
# 	def compute_lr(self, lr, lr0, tick, info=None):
# 		return self.factor * lr
#
#
# @fig.Component('scheduler/multi-step')
# class MultiStepScheduler(StepScheduler):
# 	def __init__(self, A, **kwargs):
#
# 		super().__init__(A, **kwargs)
#
# 		self.freq = None
#
# 		num = A.pull('num-steps', 1)
# 		limit = self.stop
# 		if limit is None:
# 			limit = A.pull('limit')
#
# 		size = limit // num
# 		self.steps = set([size*step for step in range(1,num)])
#
# 	def state_dict(self):
# 		data = super().state_dict()
# 		data['steps'] = self.steps
# 		return data
#
# 	def check(self, tick, info=None):
# 		return super().check(tick, info=info) and tick in self.steps
#
#
# class FunctionScheduler(Scheduler):
# 	def __init__(self, A, **kwargs):
# 		super().__init__(A, **kwargs)
#
# 		assert self.min_lr is not None, 'must have a min for exponential scheduler'
#
# 		self.limit = A.pull('limit', '<>stop')
#
# 		self.power = A.pull('power', 1)
# 		self.constrain_progress = A.pull('constrain-progress', False)
#
#
# 	def state_dict(self):
# 		data = super().state_dict()
# 		data.update({'limit':self.limit, 'power':self.power,
# 		             'constrain_progress':self.constrain_progress})
# 		return data
#
# 	def compute_lr(self, lr, lr0, tick, info=None):
# 		progress = (tick/self.limit) ** self.power
# 		if self.constrain_progress:
# 			progress = max(0.,min(progress,1.))
# 		return self._func(progress, lr0)
#
# 	def _func(self, progress, lr0):
# 		raise NotImplementedError
#
#
# @fig.Component('scheduler/cos')
# class CosScheduler(FunctionScheduler):
# 	def _func(self, progress, lr0):
# 		return (lr0-self.min_lr) * (math.cos(math.pi * progress) + 1) / 2 + self.min_lr
#
#
# @fig.Component('scheduler/exp')
# class ExpScheduler(FunctionScheduler):
# 	def _func(self, progress, lr0):
# 		return lr0 * (self.min_lr/lr0) ** progress
#
#
# @fig.Component('scheduler/lin')
# class LinScheduler(FunctionScheduler):
# 	def _func(self, progress, lr0):
# 		return lr0 - (lr0-self.min_lr) * progress




## region old

class RunningNormalization(fm.FunctionBase):
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

# endregion
