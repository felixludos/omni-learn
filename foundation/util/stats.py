
import numpy as np

import torch
from collections import deque, OrderedDict

import omnifig as fig

from .features import Configurable, Switchable

# class StatsCollector(object):
# 	def __init__(self, require_all_keys=True):
# 		assert require_all_keys
# 		self.bag = None
# 		self.steps = deque()
#
# 	def update(self, step, stats):
# 		if self.bag is None:
# 			self.bag = {stat:deque() for stat in stats}
# 		self.steps.append(step)
# 		for stat, val in stats.items():
# 			self.bag[stat].append(val)
#
# 	def export(self):
# 		data = {'steps':np.array(self.steps)}
# 		if self.bag is not None:
# 			data['stats'] = {stat:np.array(data) for stat, data in self.bag.items()}
# 		return data
#
# 	def save(self, path):
# 		torch.save(self.export(), path)


# def override_all_stats_tau(tau): # TODO: track all stats objs and then update tau
# 	pass




# @fig.AutoModifier('reg-stats')
# class RegStats(StatsContainer):
# 	def __init__(self, A, **kwargs):
#
# 		super().__init__(A, **kwargs)
#
# 		self._init_stats(A)
#
# 	def _init_stats(self, A=None):
# 		if A is not None:
# 			A.push('stats._type', 'stats', overwrite=False, silent=True)
# 			self.stats = A.pull('stats')
#
# 			if A.pull('_reg_stats', True):
# 				records = A.pull('records', None, ref=True)
# 				if records is not None:
# 					stats_fmt = A.pull('_stats_fmt', None)
# 					records.register_stats_client(self, fmt=stats_fmt)

@fig.Component('stats')
class StatsMeter(Configurable, OrderedDict):
	def __init__(self , A=None, meter_info=None, **kwargs):
		
		if meter_info is None:
			assert A is not None, 'need config to know how to create new meters'
			if 'meter-info' not in A:
				A.push('meter-info._type', 'meter', overwrite=False)
			meter_info = A.pull('meter-info', raw=True)
		
		super().__init__(A, **kwargs)
		
		self.meter_info = meter_info
		
	def set_tau(self, tau):
		for meter in self.values():
			meter.tau = tau

	def reset(self):
		for meter in self.values():
			meter.reset()
	
	# def __deepcopy__(self):
	# 	'''returns deepcopy of self'''
	# 	new = self.__class__(meter_info=self.meter_info)
	# 	for name, meter in self.items():
	# 		new[name] = meter.copy()
	# 	return new

	def _create_meter(self, name=None, tau=None):
		meter = self.meter_info.pull_self(silent=True)
		if tau is not None:
			meter.set_tau(tau)
		return meter

	def new(self, *names, tau=None):
		for name in names:
			if name not in self:
				self[name] = self._create_meter(name, tau=tau)

	def discard(self, *names):
		for name in names:
			if name in self:
				del self[name]
	def remove(self, *names):
		for name in names:
			del self[name]

	def mete(self, name, value, n=1):
		# assert isinstance(value, (int,float)) or value.size == 1, 'unknown: {} {}'.format(value.shape, value)
		self[name].mete(value, n=n)
	
	def __add__(self, other):
		new = self.copy()
		new.join(other)
		return new
	
	def join(self, other, intersection=False, fmt=None):
		'''other is another stats meter'''
		for name, meter in other.items():
			if fmt is not None:
				name = fmt.format(name)
			if name in self:
				self[name].join(meter)
			elif not intersection:
				self[name] = meter.copy()

	def shallow_join(self, other, fmt=None):
		for name, meter in other.items():
			if fmt is not None:
				name = fmt.format(name)
			self[name] = meter
	
	def vals(self, fmt='{}'):
		return {fmt.format(k):v.val.item() for k,v in self.items()}

	def avgs(self, fmt='{}'):
		return {fmt.format(k):v.avg.item() for k,v in self.items() if v.count > 0}
	
	def smooths(self, fmt='{}'):
		return {fmt.format(k): (v.smooth.item()) for k,v in self.items() if v.smooth is not None}

	def split(self):
		all_vals = {k:v.export() for k,v in self.__dict__.items()}
		key = next(iter(self.__dict__.keys()))
		try:
			shape = all_vals[key]['val'].shape
		except Exception as e:
			raise e

		stats = [{stat:{k:v[idx] for k,v in vals.items()} for stat,vals in all_vals.items()} for idx in np.ndindex(*shape)]

		return [StatsMeter(**info) for info in stats]

	def load(self, stats):
		self.update({stat:self._create_meter(stat).load(vals) for stat,vals in stats.items()})

	def export(self):
		return {name : meter.export() for name, meter in self.items()}
	#
	# def save(self, filename):
	# 	torch.save(self.export() ,filename)
	#
	# def load_dict(self, d):
	# 	for name, meter in d.items():
	# 		self._stats[name] = meter
	#
	# def load(self, filename):
	# 	self.load_dict(torch.load(filename))
	
	def __repr__(self):
		return self.__str__()
		
	def __str__(self):
		return 'StatsMeter({})'.format(', '.join(['{}:{:.4f}'.format(k,v.val.item()) for k,v in self.items()]))


@fig.Component('stats-manager')
class StatsManager(Switchable, StatsMeter):

	def __init__(self, A=None, collections=None, collection_fmts=None, **kwargs):
		
		if collection_fmts is None and A is not None:
			collection_fmts = A.pull('stat-collection-fmts', {})
		if collections is None:
			collections = {}
		
		reset_collections = set(A.pull('reset-modes', []))
		
		super().__init__(A=A, **kwargs)
		self._collections = collections
		self._reset_collections = reset_collections
		self._master_names = set()
		self._timestep = None
		self._archive = {}
		self._collection_fmts = collection_fmts
	
	def vals(self, fmt=None):
		if fmt is None:
			fmt = self._collection_fmts.get(self.get_mode(), '{}')
		return super().vals(fmt=fmt)
	
	def avgs(self, fmt=None):
		if fmt is None:
			fmt = self._collection_fmts.get(self.get_mode(), '{}')
		return super().avgs(fmt=fmt)
	
	def smooths(self, fmt=None):
		if fmt is None:
			fmt = self._collection_fmts.get(self.get_mode(), '{}')
		return super().smooths(fmt=fmt)
	
	def new(self, *names, tau=None):
		self._master_names.update(names)
		super().new(*names, tau=tau)
	
	def remove(self, *names):
		self._master_names.difference_update(names)
		super().remove(*names)

	def discard(self, *names):
		self._master_names.difference_update(names)
		super().discard(*names)
		
	def _create_collection(self):
		return {name:self._create_meter(name) for name in self._master_names}
	
	def set_step(self, ticks):
		self._timestep = ticks
	
	def archive(self, ticks=None, mode=None):
		if mode is None:
			mode = self.get_mode()
		if ticks is None:
			ticks = self._timestep
		if mode is None or ticks is None:
			return
		
		self._sync_collections()
		
		if mode not in self._archive:
			self._archive[mode] = {}
		self._archive[mode][str(ticks)] = {name: meter.export() for name, meter in self._collections[mode].items()}
	
	def switch_to(self, mode):
		
		old = self.get_mode()
		if mode == old:
			return
		
		self.archive()
		
		super().switch_to(mode)
		self.clear()
		
		if mode not in self._collections or mode in self._reset_collections:
			self._collections[mode] = self._create_collection()
		self.update(self._collections[mode])
		
	def _sync_collections(self):
		mode = self.get_mode()
		if mode is not None:
			self._collections[self.get_mode()] = dict(self)
		
	def export(self):
		self.archive()
		return self._archive
		
	def load(self, stats):
		mode = None
		self._archive = stats
		self._collections = {}
		for mode, arc in stats.items():
			last = arc[str(max(map(int,arc.keys())))]
			if mode not in self._reset_collections:
				self._collections[mode] = {name:self._create_meter(name).load(vals)
				                                for name, vals in last.items()}
		
		if mode is not None:
			self._master_names = set(last.keys())
		
		self.clear()
		self.mode = None


class StatsClient(Configurable):
	def __init__(self, A, **kwargs):
		stats = A.pull('stats', None, ref=True)
		if stats is None:
			print('WARNING: no stats manager found')
		fmt = A.pull('stats-fmt', None)
		
		default_tau = A.pull('smooth-tau', '<>tau', None)
		
		super().__init__(A, **kwargs)
		
		self._stats = stats
		self._stats_fmt = fmt
		self._stats_tau = default_tau
	
	def register_stats(self, *names):
		if self._stats is not None:
			if self._stats_fmt is not None:
				names = [self._stats_fmt.format(name) for name in names]
			self._stats.new(*names, tau=self._stats_tau)
	
	def mete(self, name, val, n=1):
		if self._stats is not None:
			if self._stats_fmt is not None:
				name = self._stats_fmt.format(name)
			self._stats.mete(name, val, n=n)
	
	def get_stat(self, name):
		if self._stats is not None:
			if self._stats_fmt is not None:
				name = self._stats_fmt.format(name)
			return self._stats.get(name, None)


_tau = 0.005
def set_default_tau(tau):
	global _tau
	_tau = tau

### Computes sum/avg stats
@fig.Component('meter')
class AverageMeter(Configurable):
	"""Computes and stores the average and current value"""
	def __init__(self, A=None, tau=None, **kwargs):
		super().__init__(A, **kwargs)
		
		self.reset()
		
		if tau is None:
			assert A is not None, 'no config provided'
			tau = A.pull('smooth-tau', '<>tau', _tau)
		self.set_tau(tau)
		
	def set_tau(self, tau):
		self.tau = torch.tensor(tau)
		
	def copy(self):
		new = self.__class__(tau=self.tau)
		for k, v in self.__dict__.items():
			if k == 'tau':
				new.tau = self.tau
				continue
			try:
				new.__dict__[k] = None if v is None else v.clone()
			except Exception as e:
				print(k, type(v), v)
				raise e
		return new
	
	def reset(self):
		self.val = torch.tensor(0.)
		self.avg = torch.tensor(0.)
		self.sum = torch.tensor(0.)
		self.count = torch.tensor(0.)
		self.max = None
		self.min = None
		self.smooth = None
		self.var = torch.tensor(0.)
		self.std = torch.tensor(0.)
		self.S = torch.tensor(0.)
	
	def mete(self, val, n=1):
		try:
			val = val.float().detach().cpu()
		except:
			val = torch.tensor(val).float()
		self.val = val
		self.sum += self.val * n
		prev_count = self.count
		self.count += n
		delta = self.val - self.avg
		self.S += delta**2 * n * prev_count / self.count
		self.avg = self.sum / self.count
		self.max = self.val if self.max is None else np.maximum(self.max, val)
		self.min = self.val if self.min is None else np.minimum(self.min, val)
		self.var = self.S / self.count
		self.std = torch.sqrt(self.var)
		self.smooth = self.val if self.smooth is None else (self.smooth*(1-self.tau) + self.val*self.tau)

	def __len__(self):
		return self.count.item()

	def load(self, vals):
		for k, v in vals.items():
			self.__dict__[k] = torch.tensor(v)
		return self
	
	def export(self): #
		if self.val.nelement() > 1:
			return {k: v.detach().cpu().numpy() for k, v in self.__dict__.items() if v is not None}
		else:
			return {k: v.item() for k,v in self.__dict__.items() if v is not None}

	def type(self):
		return '{}-{}'.format(self.val.type(), tuple(self.val.size()))
		
	def join(self, other):
		'''other is another average meter, assumed to be more uptodate (for val)
			this does not mutate other'''
		self.val = other.val
		if self.count+other.count == 0: # both are empty
			self.reset()
			return

		delta = other.avg - self.avg
		prev_count = self.count
		self.count += other.count
		self.S += other.S + delta ** 2 * other.count * prev_count / self.count
		self.var = self.S / self.count
		self.std = np.sqrt(self.var)

		self.sum += other.sum
		self.avg = self.sum / self.count

		if self.max is None:
			self.max = other.max
		elif other.max is not None:
			self.max = max(self.max, other.max)
		if self.min is None:
			self.min = other.min_val
		elif other.min_val is not None:
			self.min = min(self.min, other.min_val)

