
import sys, os
import time
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import scipy.misc
from io import BytesIO  # Python 3.x
from PIL import Image

import omnifig as fig

@fig.AutoComponent('logger')
class Logger(object):

	def __init__(self, log_dir=None, tensorboard=False, txt=False, logfile=None, **kwargs):

		self.tblog = None
		if tensorboard:
			try:
				self.tblog = SummaryWriter(log_dir=log_dir, **kwargs)
			except:
				self.tblog = None

		self.txtlog = None
		if txt:
			if logfile is None:
				logfile = open(os.path.join(log_dir, 'logfile.txt'), 'a+')
			self._old_stdout = sys.stdout
			self.txtlog = Tee(self._old_stdout, logfile)
			sys.stdout = self.txtlog

			now = time.strftime("%b-%d-%Y-%H%M%S")
			title = '**** Beginning Log {} ****\n'.format(now)
			title_stars = '*' *(len(title ) -1) + '\n'

			self.txtlog.write(title_stars + title + title_stars, logonly=True)

		self.global_step = None
		self.tag_fmt = None
		

	def set_step(self, step):
		self.global_step = step

	def get_step(self):
		return self.global_step

	def set_tag_format(self, fmt=None):
		self.tag_fmt = fmt
		
	def get_tag_format(self):
		return self.tag_fmt

	def add_hparams(self, param_dict, metrics={}):
		self.tblog.add_hparams(param_dict, metrics)

	def add(self, data_type, tag, *args, global_step=None, **kwargs): # TODO: test and maybe clean

		if self.tblog is None:
			return None

		# if data_type == 'scalar' and not isinstance(args[0], float):
		# 	print(tag, type(args[0]), args[0])
		# 	assert False

		add_fn = self.tblog.__getattribute__('add_{}'.format(data_type))

		if global_step is None:
			global_step = self.global_step

		if self.tag_fmt is not None:
			tag = self.tag_fmt.format(tag)

		add_fn(tag, *args, global_step=global_step, **kwargs)

	def flush(self):
		if self.tblog is not None:
			self.tblog.flush()
		if self.txtlog is not None:
			self.txtlog.flush()


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





class Old_Logger(object):
	def __init__(self, log_dir, tensorboard=True, txt=False, auto_step=False, step0=0, step_delta=1, logfile=None):
		raise Exception('Deprecated, use Logger instead')
		now = time.strftime("%b-%d-%Y-%H%M%S")

		self.tblog = TBLogger(log_dir) if tensorboard else None

		if txt:
			if logfile is None:
				logfile = open(os.path.join(log_dir, 'logfile.txt'), 'a+')
			self._old_stdout = sys.stdout
			self.txtlog = Tee(self._old_stdout, logfile)

			title = '**** Beginning Log {} ****\n'.format(now)
			title_stars = '* ' *(len(title ) -1) + '\n'

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

		for k ,v in info.items():
			self.tblog.scalar_summary(k, v, step)

	def update_images(self, info, step=None): # input 1 x H x W x 3 numpy images
		if self.tblog is None:
			return

		if step is None:
			step = self.step

		for k ,v in info.items():
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
		hist.min_val = float(np.min_val(values))
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


