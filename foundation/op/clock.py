import time
from collections import OrderedDict

from tqdm import tqdm, tqdm_notebook

# from omnibelt import InitSingleton
import omnifig as fig

from .. import util

################
# region Clocks
################

@fig.Component('clock')
class Clock:
	def __init__(self, A, **kwargs):
		self._ticks = 0
		self._alerts = OrderedDict()
		self._info = None
	
	def get_info(self):
		return self._info
	
	def set_info(self, info):
		self._info = info
	
	def register_alert(self, name, alert, **unused):
		if name is None:
			name = f'{alert}#{id(alert)}'
		self._alerts[name] = alert
	
	def _call_alert(self, name, alert, info=None):
		if info is None:
			info = self._info
		return alert.activate(self._ticks, info)
	
	def get_time(self):
		return self._ticks
	
	def set_time(self, ticks):
		self._ticks = ticks
	
	def tick(self, info=None):
		for name, alert in self._alerts.items():
			if alert.check(self._ticks, info):
				self._call_alert(name, alert, info=info)
		self._ticks += 1
	
	# region unused

	def init_pbar(self, limit=None, **kwargs):
		pass
	
	def set_description(self, desc):
		pass
	
	# endregion

	pass

@fig.AutoModifier('clock-limit')
class Limited(Clock):
	def __init__(self, A, **kwargs):
		limit = A.pull('limit', None)
		super().__init__(A, **kwargs)
		self.limit = limit

	def get_limit(self):
		return self.limit
	
	def set_limit(self, limit):
		old = self.limit
		self.limit = limit
		return old

@fig.AutoModifier('clock-stats')
class Stats(util.RegStats, Clock):
	def register_alert(self, name, alert, add_to_stats=True, **unused):
		if add_to_stats:
			self.stats.new(name)
		return super().register_alert(name, alert, **unused)

	def _call_alert(self, name, alert, info=None):
		val = super()._call_alert(name, alert, info=info)
		if val is not None:
			self.stats.update(name, val)
		return val

@fig.AutoModifier('clock-timed')
class Timed(Stats):
	def __init__(self, A):
		super().__init__(A)
		self._timed_fmt = A.pull('timed-fmt', 'time-{}')
		self._timed_stats = {}
	
	def register_alert(self, name, alert, add_to_stats=True, add_to_timed=True):
		if add_to_timed:
			key = self._timed_fmt.format(name)
			self._timed_stats[name] = key
			self.stats.new(key)
		super().register_alert(name, alert, add_to_stats=add_to_stats)

	def _call_alert(self, name, alert, info=None):
		start = time.time()
		
		out = super()._call_alert(name, alert, info=info)
		
		if name in self._timed_stats:
			self.stats.update(self._timed_stats[name], time.time() - start)
			
		return out

@fig.AutoModifier('clock-pbar')
class Pbar(Limited):
	def __init__(self, A, **kwargs):
		
		A.push('pbar._type', 'progress-bar', silent=False, overwrite=False)
		pbar = A.pull('pbar')
		
		super().__init__(A, **kwargs)
		
		self.pbar = pbar
		
	def init_pbar(self, limit=None, **kwargs):
		if limit is None:
			limit = self._limit
		self.pbar.init_pbar(limit, **kwargs)
		
	def reset(self):
		self.pbar.reset()
		
	def set_limit(self, limit):
		old = super().set_limit(limit)
		if self.pbar is not None and old is None or old != limit:
			self.init_pbar()
		return old
		
	def tick(self, info=None):
		out = super().tick(info=info)
		if self.pbar is not None:
			self.pbar.update(n=1)
		return out
		
	def set_description(self, desc):
		self.pbar.set_description(desc)
	

# endregion
################


################
# region Alerts
################

class Alert:
	def __init__(self, A=None, **kwargs):
		pass
	
	def check(self, tick, info=None):
		return True
	
	def activate(self, tick, info=None):
		'''
		
		:param tick: int
		:param info: object passed to clock.tick()
		:return: new value (representative of the alert) or none
		'''
		pass

class CustomAlert(Alert):
	def __init__(self, activate=None, check=None, **kwargs):
		super().__init__(**kwargs)
		self._activate = activate
		self._check = check
		
	def check(self, tick, info=None):
		if self._check is None:
			return True
		return self._check(tick, info=info)

	def activate(self, tick, info=None):
		if self._activate is None:
			pass
		return self._activate(tick, info=info)

@fig.AutoModifier('alert/named')
class Named(Alert):
	def __init__(self, A, **kwargs):
		ident = A.pull('ident', '<>name', None)
		super().__init__(A, **kwargs)
		self.name = ident
		
	def get_name(self):
		return self.name


@fig.AutoModifier('alert/reg')
class Reg(Named):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		if 'clock' not in A:
			A.push('clock._type', 'clock', overwrite=False, silent=True)
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
		self.zero = zero
	
	def check(self, tick, info=None):
		return self.freq is None or ((self.zero or self.freq >= 1) and tick % self.freq == 0)

# endregion
################