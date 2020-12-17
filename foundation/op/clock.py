import time
from collections import OrderedDict

from tqdm import tqdm, tqdm_notebook

# from omnibelt import InitSingleton
import omnifig as fig

from .runs import Clock, Alert
from .. import util

################
# region Clocks
################


@fig.AutoModifier('clock-limit')
class Limited(Clock):
	def __init__(self, A, **kwargs):
		limit = A.pull('limit', None)
		super().__init__(A, **kwargs)
		self.limit = limit

	def get_limit(self):
		return self.limit
	
	def get_remaining(self):
		lim = self.get_limit()
		if lim is not None:
			return max(lim-self.get_time(),0)
	
	def set_limit(self, limit):
		old = self.limit
		self.limit = limit
		return old
	
	def step(self, info=None, n=None):
		if n is None:
			n = self.get_remaining()
		return super().step(info=info, n=n)


# @fig.AutoModifier('clock-pbar')
# class Pbar(Limited):
# 	def __init__(self, A, **kwargs):
#
# 		A.push('pbar._type', 'progress-bar', silent=False, overwrite=False)
# 		pbar = A.pull('pbar')
#
# 		super().__init__(A, **kwargs)
#
# 		self.pbar = pbar
#
# 	def init_pbar(self, limit=None, **kwargs):
# 		if limit is None:
# 			limit = self._limit
# 		self.pbar.init_pbar(limit, **kwargs)
#
# 	def reset(self):
# 		self.pbar.reset()
#
# 	def set_limit(self, limit):
# 		old = super().set_limit(limit)
# 		if self.pbar is not None and old is None or old != limit:
# 			self.init_pbar()
# 		return old
#
# 	def tick(self, info=None):
# 		out = super().tick(info=info)
# 		if self.pbar is not None:
# 			self.pbar.update(n=1)
# 		return out
#
# 	def set_description(self, desc):
# 		self.pbar.set_description(desc)


@fig.AutoModifier('clock/stats')
class Stats(util.StatsClient, Clock):
	def register_alert(self, name, alert, add_to_stats=True, **unused):
		if add_to_stats:
			self.register_stats(name)
		return super().register_alert(name, alert, **unused)
	
	def _call_alert(self, name, alert, info=None):
		val = super()._call_alert(name, alert, info=info)
		if val is not None:
			self.update_stat(name, val)
		return val


@fig.AutoModifier('clock/timed')
class Timed(Stats):
	def __init__(self, A):
		super().__init__(A)
		self._timed_fmt = A.pull('timed-fmt', 'time-{}')
		self._timed_stats = {}
	
	def register_alert(self, name, alert, add_to_stats=True, add_to_timed=True):
		if add_to_timed:
			key = self._timed_fmt.format(name)
			self._timed_stats[name] = key
			self.register_stats(key)
		super().register_alert(name, alert, add_to_stats=add_to_stats)
	
	def _call_alert(self, name, alert, info=None):
		start = time.time()
		
		out = super()._call_alert(name, alert, info=info)
		
		if name in self._timed_stats:
			self.stats.update(self._timed_stats[name], time.time() - start)
		
		return out

# endregion
################


################
# region Alerts
################





@fig.AutoModifier('alert/named')
class Named(Alert):
	def __init__(self, A, **kwargs):
		ident = A.pull('ident', '<>__origin_key', None)
		super().__init__(A, **kwargs)
		self.name = ident
		
	def get_name(self):
		return self.name


@fig.AutoModifier('alert/priority')
class Priority(Alert):
	def __init__(self, A, **kwargs):
		priority = A.pull('priority', None)
		super().__init__(A, **kwargs)
		self.priority = priority
		
	def get_priority(self):
		return self.priority


@fig.AutoModifier('alert/reg')
class Reg(Named):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		clock = A.pull('clock', None, ref=True)
		if clock is not None:
			clock.register_alert(self.get_name(), self)


@fig.AutoModifier('alert/freq')
class Freq(Alert):
	def __init__(self, A, **kwargs):
		zero = A.pull('include-zero', False)
		freq = A.pull('freq', None)
		super().__init__(A, **kwargs)
		self.freq = freq
		self._include_zero = zero
	
	def check(self, tick, info=None):
		return self.freq is None or ((self._include_zero or self.freq >= 1) and tick % self.freq == 0)

# endregion
################