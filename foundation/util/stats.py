import sys, os
import time
import numpy as np
import tensorflow as tf
import scipy.misc
from io import BytesIO  # Python 3.x
import torch
from collections import deque
from PIL import Image

class Logger(object):
	def __init__(self, log_dir, tensorboard=True, txt=False, auto_step=False, step0=0, step_delta=1, logfile=None):

		now = time.strftime("%b-%d-%Y-%H%M%S")

		self.tblog = TBLogger(log_dir) if tensorboard else None

		if txt:
			if logfile is None:
				logfile = open(os.path.join(log_dir, 'logfile.txt'), 'a+')
			self._old_stdout = sys.stdout
			self.txtlog = Tee(self._old_stdout, logfile)

			title = '**** Beginning Log {} ****\n'.format(now)
			title_stars = '*'*(len(title)-1) + '\n'

			self.txtlog.write(title_stars + title + title_stars, logonly=True)

		self.auto_step = auto_step
		self.step = step0
		self.delta = step_delta

	def update(self, info, step=None):
		if self.tblog is None:
			return

		if step is None:
			step = self.step
		self.step += self.delta

		if isinstance(info, StatsMeter):
			info = {name: info[name].val.item() for name in info.keys()}

		for k,v in info.items():
			self.tblog.scalar_summary(k, v, step)
			
	def update_images(self, info, step=None): # input 1 x H x W x 3 numpy images
		if self.tblog is None:
			return

		if step is None:
			step = self.step

		for k,v in info.items():
			self.tblog.image_summary(k, v, step)
		

class TBLogger(object):
	def __init__(self, log_dir):
		"""Create a summary writer logging to log_dir."""
		self.writer = tf.summary.FileWriter(log_dir +'/')
		self.tag_steps = {}
	
	def scalar_summary(self, tag, value, step=None):
		"""Log a scalar variable."""
		if step is None:
			if tag not in self.tag_steps:
				self.tag_steps[tag] = 0
			self.tag_steps[tag] += 1
			step = self.tag_steps[tag]
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
		self.writer.add_summary(summary, step)
	
	def image_summary(self, tag, images, step):
		"""Log a list of images."""
		
		img_summaries = []
		for i, img in enumerate(images):
			# Write the image to a string
			s = BytesIO()
			Image.fromarray(img).save(s, format="jpeg")
			
			# Create an Image object
			img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
									   height=img.shape[0],
									   width=img.shape[1])
			# Create a Summary value
			img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
		
		# Create and write Summary
		summary = tf.Summary(value=img_summaries)
		self.writer.add_summary(summary, step)
	
	def histo_summary(self, tag, values, step, bins=1000):
		"""Log a histogram of the tensor of values."""
		
		# Create a histogram using numpy
		counts, bin_edges = np.histogram(values, bins=bins)
		
		# Fill the fields of the histogram proto
		hist = tf.HistogramProto()
		hist.min = float(np.min(values))
		hist.max = float(np.max(values))
		hist.num = int(np.prod(values.shape))
		hist.sum = float(np.sum(values))
		hist.sum_squares = float(np.sum(values ** 2))
		
		# Drop the start of the first bin
		bin_edges = bin_edges[1:]
		
		# Add bin edges and counts
		for edge in bin_edges:
			hist.bucket_limit.append(edge)
		for c in counts:
			hist.bucket.append(c)
		
		# Create and write Summary
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
		self.writer.add_summary(summary, step)
		self.writer.flush()

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

### Write to stdout and log file
class Tee(object):
	def __init__(self, stdout, logfile):
		self.stdout = stdout
		self.logfile = logfile
	
	def write(self, obj, logonly=False):
		self.logfile.write(obj)
		if logonly:
			return
		self.stdout.write(obj)
	
	def flush(self):
		self.stdout.flush()
	
	def __del__(self):
		self.logfile.close()