
import numpy as np

import torch
from collections import deque



class StatsCollector(object):
	def __init__(self, require_all_keys=True):
		assert require_all_keys
		self.bag = None
		self.steps = deque()

	def update(self, step, stats):
		if self.bag is None:
			self.bag = {stat:deque() for stat in stats}
		self.steps.append(step)
		for stat, val in stats.items():
			self.bag[stat].append(val)

	def export(self):
		data = {'steps':np.array(self.steps)}
		if self.bag is not None:
			data['stats'] = {stat:np.array(data) for stat, data in self.bag.items()}
		return data

	def save(self, path):
		torch.save(self.export(), path)

class StatsMeter(object):
	def __init__(self , *names, tau=0.001, **stats):
		self._stats = {}
		
		self.tau = tau
		
		for name in names:
			self.new(name)
		if len(stats):
			self.load(stats)

	def set_tau(self, tau):
		self.tau = tau
		for s in self._stats.values():
			s.tau = tau

	def reset(self):
		for stat in self._stats.values():
			stat.reset()
	
	def copy(self):
		'''returns deepcopy of self'''
		new = StatsMeter(tau=self.tau)
		for name, meter in self._stats.items():
			new._stats[name] = meter.copy()
		return new
	
	def __getitem__(self, name):
		#assert name in self._stats, name + ' not found'
		return self._stats[name]

	def __contains__(self, name):
		return name in self._stats

	def new(self, *names):
		for name in names:
			#assert name not in self._stats, 'The stat ' + name + ' already exists'
			if name not in self._stats:
				self._stats[name] = AverageMeter(tau=self.tau)
	
	def update(self, name, value, n=1):
		self._stats[name].update(value, n=n)
	
	def __add__(self, other):
		new = self.copy()
		new.join(other)
		return new
	
	def join(self, other, intersection=False, prefix=''):
		'''other is another stats meter'''
		for name, meter in other._stats.items():
			name = prefix + name
			if name in self._stats:
				self._stats[name].join(meter)
			elif not intersection:
				self._stats[name] = meter.copy()

	def shallow_join(self, other, prefix=''):
		for name, meter in other._stats.items():
			name = prefix + name
			self._stats[name] = meter
	
	def keys(self):
		return self._stats.keys()
	
	def items(self):
		return self._stats.items()

	def vals(self, fmt='{}'):
		return {fmt.format(k):v.val.item() for k,v in self._stats.items()}

	def avgs(self, fmt='{}'):
		return {fmt.format(k):v.avg.item() for k,v in self._stats.items()}
	
	def smooths(self, fmt='{}'):
		return {fmt.format(k): (v.smooth.item() if v.smooth is not None else float('nan')) for k,v in self._stats.items()}

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
		self._stats.update({stat:AverageMeter(vals) for stat,vals in stats.items()})

	# def export(self):
	# 	return {name :meter for name, meter in self._stats.items()}
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
		return 'StatsMeter({})'.format(', '.join(['{}:{:.4f}'.format(k,v.val.item()) for k,v in self._stats.items()]))

### Computes sum/avg stats
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, vals=None, tau=0.001):
		if vals is not None:
			self.load(vals)
		else:
			self.reset()
			
		self.tau = tau
	
	def copy(self):
		new = AverageMeter()
		for k, v in self.__dict__.items():
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
	
	def update(self, val, n=1):
		try:
			val = val.float().detach().cpu()
		except:
			pass
		self.val = torch.tensor(val).float()
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
			self.min = other.min
		elif other.min is not None:
			self.min = min(self.min, other.min)

