
from collections import OrderedDict

# from omnibelt import InitSingleton
import omnifig as fig

from .. import util

################
# region Clocks
################

@fig.Component('clock')
class Clock:
	def __init__(self, A):
		self._ticks = 0
		self._alerts = OrderedDict()
	
	def register_alert(self, name, alert):
		self._alerts[name] = alert
	
	def _call_alert(self, name, alert, info=None):
		return alert.activate(self._ticks, info)
	
	def get_time(self):
		return self._ticks
	
	def tick(self, info=None):
		self._ticks += 1
		for name, alert in self._alerts.items():
			if alert.check(self._ticks, info):
				self._call_alert(name, alert, info=info)


@fig.AutoModifier('clock-stats')
class Stats(Clock):
	def __init__(self, A):
		A.push('stats._type', 'stats', overwrite=False, silent=True)
		super().__init__(A)
		self.stats = A.pull('stats')
	
	def register_alert(self, name, alert, add_to_stats=True):
		if add_to_stats:
			self.stats.new(name)
		return super().register_alert(name, alert)

	def _call_alert(self, name, alert, info=None):
		val = super()._call_alert(name, alert, info=info)
		if val is not None:
			self.stats.update(name, val)
		return val

# endregion
################


################
# region Alerts
################

class Alert:
	def __init__(self, A):
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


@fig.AutoModifier('alert/named')
class Named(Alert):
	def __init__(self, A):
		ident = A.pull('ident', '<>name', None)
		super().__init__(A)
		self.name = ident
		
	def get_name(self):
		return self.name


@fig.AutoModifier('alert/reg')
class Reg(Named):
	def __init__(self, A):
		if 'clock' not in A:
			A.push('clock._type', 'clock', overwrite=False, silent=True)
		clock = A.pull('clock', ref=True)
		super().__init__(A)
		clock.register_alert(self.get_name(), self)


@fig.AutoModifier('alert/freq')
class Freq(Alert):
	def __init__(self, A):
		freq = A.pull('freq', None)
		super().__init__(A)
		self.freq = freq
	
	def check(self, tick, info=None):
		return self.freq is None or (self.freq >= 1 and tick % self.freq == 0)

# endregion
################