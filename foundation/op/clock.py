
import time
from pathlib import Path
from collections import OrderedDict
from omnibelt import load_json, save_json
import omnifig as fig

from .. import util
from ..util import Named, StatsClient, Configurable, Statelike, Checkpointable

################
# region Clocks
################

class AlertNotFoundError(Exception):
	def __init__(self, name):
		super().__init__(f'Could not find an alert named {name}')
		self.name = name

@fig.Component('clock/simple')
class SimpleClock(Checkpointable, Configurable):
	def __init__(self, A, skip_load=None, **kwargs):
		
		if skip_load is None:
			skip_load = A.pull('skip_load', False)
		
		ckpt = None
		if not skip_load:
			ckpt = A.pull('_load-ckpt', None)
		
		super().__init__(A, **kwargs)
		self.ticks = 0
		self.alerts = OrderedDict()
		self._info = None
		
		self._skip_load = None
		
		if ckpt is not None:
			self.load_checkpoint(ckpt)
	
	def prep(self, info=None):
		if info is None:
			info = self._info
		for idx, (name, alert) in enumerate(self.alerts.items()):
			alert.prep(idx, info=info)
	
	def sort_alerts(self, start_with=None, end_with=None, strict=True):
		order = OrderedDict()
		
		if start_with is not None:
			for name in start_with:
				if name in self.alerts:
					order[name] = self.alerts[name]
				elif strict:
					raise AlertNotFoundError(name)
				
		for name, alert in self.alerts.items():
			if end_with is None or name not in end_with:
				order[name] = alert
		
		if end_with is not None:
			for name in end_with:
				if name in self.alerts:
					order[name] = self.alerts[name]
				elif strict:
					raise AlertNotFoundError(name)
		
		self.alerts.clear()
		self.alerts.update(order)
	
	def get_info(self):
		return self._info
	
	def set_info(self, info):
		self._info = info
	
	def register_alert_fn(self, name, check=None, activate=None):
		self.register_alert(name, CustomAlert(check=check, activate=activate))
	
	def register_alert(self, name, alert, **unused):
		if name is None:
			name = f'{alert}#{id(alert)}'
		self.alerts[name] = alert
	
	def checkpoint(self, path, ident='clock'):
		
		path = Path(path) / f'{ident}.json'
		
		states = {}
		for name, alert in self.alerts.items():
			if isinstance(alert, Savable):
				states[name] = alert.state_dict()
			
		save_json({'time': self.get_time(), 'alert_states': states}, path)
	
	def load_checkpoint(self, path, ident='clock'):
		
		path = Path(path) / f'{ident}.json'
		
		if path.is_file():
			data = load_json(path)
			
			self.ticks = data['time']
			
			states = data['alert_states']
			for name, state in states.items():
				if name in self.alerts:
					self.alerts[name].load_state_dict(state)
	
	def clear(self):
		self.alerts.clear()
		self._info = None
	
	def _call_alert(self, name, alert, info=None):
		if info is None:
			info = self._info
		return alert.activate(self.ticks, info)
	
	def get_time(self):
		return self.ticks
	
	def __len__(self):
		return self.get_time()
	
	def set_time(self, ticks):
		self.ticks = ticks
	
	def tick(self, info=None):
		self.ticks += 1
		for name, alert in self.alerts.items():
			if alert.check(self.ticks, info):
				self._call_alert(name, alert, info=info)
	
	def step(self, info=None, n=None):
		if n is None:
			n = 1
		for _ in range(n):
			self.tick(info=info)


@fig.AutoModifier('clock/limit')
class Limited(SimpleClock):
	def __init__(self, A, **kwargs):
		limit = A.pull('limit', None)
		super().__init__(A, **kwargs)
		self.limit = limit

	def get_limit(self):
		return self.limit
	
	def get_remaining(self):
		lim = self.get_limit()
		if lim is not None:
			return max(lim-self.get_time(), 0)
	
	def set_limit(self, limit):
		old = self.limit
		self.limit = limit
		return old
	
	def tick(self, info=None, force=False):
		if self.get_limit() < self.get_time() and not force:
			raise StopIteration
		return super().tick(info=info)
	
	def step(self, info=None, n=None):
		if n is None:
			n = self.get_remaining()
		return super().step(info=info, n=n)


@fig.AutoModifier('clock/stats')
class Stats(StatsClient, SimpleClock):
	# def __init__(self, A, **kwargs):
	# 	super().__init__(A, **kwargs)
	
	
	def register_alert(self, name, alert, add_to_stats=True, **unused):
		if add_to_stats:
			self.register_stats(name)
		return super().register_alert(name, alert, **unused)
	
	def _call_alert(self, name, alert, info=None):
		val = super()._call_alert(name, alert, info=info)
		if val is not None:
			self.mete(name, val)
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
			self.mete(self._timed_stats[name], time.time() - start)
		
		return out

@fig.Component('clock')
class Clock(Timed, Limited):
	pass


# endregion
################


################
# region Alerts
################


class AlertBase:
	def prep(self, order, info=None):
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


class CustomAlert(AlertBase):
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


class Alert(Configurable, AlertBase):
	pass



@fig.AutoModifier('alert/reg')
class Reg(Named, Alert):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		clock = A.pull('clock', None, ref=True)
		if clock is not None:
			clock.register_alert(self.get_name(), self)


@fig.AutoModifier('alert/freq')
class Freq(Alert):
	def __init__(self, A, **kwargs):
		zero = A.pull('include_zero', False)
		freq = A.pull('freq', None)
		super().__init__(A, **kwargs)
		self.freq = freq
		self._include_zero = zero
	
	def check(self, tick, info=None):
		return self.freq is None or ((self._include_zero or tick >= 1) and tick % self.freq == 0)

class Savable(Statelike, Alert):
	pass

# endregion
################