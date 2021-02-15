
import sys, os, shutil, time
import traceback
import numpy as np
import torch
import yaml
from torch import multiprocessing as mp

from omnibelt import get_printer
import omnifig as fig

from bisect import bisect_left

from .. import util

# import tensorflow as tf
# from tensorboard import main as tb
from tensorboard import program
import pandas as pd

prt = get_printer(__name__)

try:
	import matplotlib.pyplot as plt
	from matplotlib.figure import Figure
	from matplotlib.animation import Animation
	
except ImportError:
	prt.warning('matplotlib not found')
	
try:
	import seaborn as sns
except ImportError:
	prt.warning('seaborn not found')

try:
	from IPython.display import HTML
except ImportError:
	prt.warning('ipython not found')

class Visualization(object): # wriapper
	def __init__(self, obj):
		self.obj = obj

	def view(self, scale=1):

		if isinstance(self.obj, util.Video):
			return HTML(self.obj.as_animation(scale=scale).to_html5_video())

		return self.obj

	def __str__(self):
		return 'Viz({})'.format(self.obj)

	def __repr__(self):
		return repr(self.obj)

	def save(self, path, vid_ext='mp4', img_ext='png', **kwargs):
		if isinstance(self.obj, Animation):
			if 'fps' not in kwargs:
				kwargs['fps'] = 30
			if 'writer' not in kwargs:
				kwargs['writer'] = 'imagemagick'

			if 'gif' not in path and 'mp4' not in path:
				path = '{}.{}'.format(path, vid_ext)

			self.obj.save(path, **kwargs)

		elif isinstance(self.obj, Figure):
			if '.pdf' not in path and '.png' not in path: # only png or pdf
				path = '{}.{}'.format(path, img_ext)
			self.obj.savefig(path)

		elif isinstance(self.obj, util.Video):

			if '.gif' not in path and '.mp4' not in path: # only gif or mp4
				path = '{}.{}'.format(path, vid_ext)

			self.obj.export(path)

		else:
			raise Exception('Unknown obj: {}'.format(type(self.obj)))


class Run_Manager(object):
	def __init__(self, names, root=None, override=None, pbar=None):
		
		self._names = names
		self._root = root
		self._override = override
		self._pbar = pbar
		
		self._load_runs()
		
	def reset(self):
		self._runs = []
		self.active = self._runs
		
	def _load_runs(self):
		self.reset()
		kwargs = {}
		if self._root is not None:
			kwargs['saveroot'] = str(self._root)
		if self._override is not None:
			kwargs['override'] = self._override.copy()
		todo = [None]*len(self._names), self._names
		if isinstance(self._names, dict):
			todo = self._names.items()
		todo = todo if self._pbar is None else self._pbar(todo)
		for title, name in todo:
			load_config = fig.get_config(load=name, **kwargs)
			load_config.set_silent(True)
			run = fig.run('load-run', load_config)
			if title is not None:
				run.ident = title
			self._runs.append(run)
		self.active = self._runs.copy()

	def __len__(self):
		return len(self.active)

	def __iter__(self):
		return iter(self.active)

	def remove(self, name):

		if name not in self:
			raise ValueError(name)

		del self[self(name).idx]

	def __delitem__(self, item):
		del self.active[item]
		self.set_active(self.active)

	def map(self, fn, safe=False, pbar=None, reduce=None):
		'''
		fn is a callable taking one run as input
		'''

		if pbar is None:
			pbar = self._pbar

		outs = []

		seq = self.active if pbar is None else pbar(self.active)

		for run in seq:
			try:
				out = fn(run)
				outs.append(out)
			except Exception as e:
				if safe:
					print('{} failed'.format(run.get_name()))
					traceback.print_exc()
				else:
					raise e

		if pbar is not None:
			seq.close()

		if reduce is not None:
			return reduce(outs)
		return outs

	def through(self, **map_kwargs):

		def _execute(fn, args=[], kwargs={}):
			return self.map(lambda run: fn(run, *args, **kwargs),
			                **map_kwargs)

		return util.make_ghost(Run, _execute)

	def clear_run_cache(self, **kwargs):
		self.map(lambda run: run.purge(), **kwargs)

	def load_results(self, ident, load_kwargs={}, **kwargs):
		return self.map(lambda run: run.get_results(ident, **load_kwargs), **kwargs)

	# def stats_dataframe(self): # gets all stats
	#
	# 	data = []
	#
	# 	keys = None
	#
	# 	for run in self.active:
	# 		if 'stats' in run:
	# 			if keys is None:
	# 				keys = list(run.stats.keys())
	#
	# 			row = [run.stats[key] for key in keys]
	#
	# 			data.append(row)
	#
	# 	if keys is None:
	# 		print('No stats found')
	# 		return None
	#
	# 	return pd.DataFrame(data, columns=keys)



	def plot_stats(self, *stats, figax=None, get_X=None, palette='muted', **kwargs):

		df = self.stats_dataframe()[list(stats)]

		if get_X is None:
			get_X = lambda run: run.idx

		X = [get_X(run) for run in self]

		df['run'] = X

		yname = 'value'
		hue = 'stat'

		if len(stats) > 1:
			df = pd.melt(df, id_vars='run', var_name=hue, value_name=yname)
		else:
			yname = stats[0]
			hue = None
			palette = None

		if figax is None:
			figax = plt.subplots()
		_, ax = figax

		sns.barplot(x='run', y=yname, hue=hue, data=df, ax=ax, palette=palette, **kwargs)

		# plt.title(', '.join(stats))

		return figax


	# def prep_info(self, checkpoint=None, load_last=True, clear_info=False, force=False, name=None):
	#
	# 	self.select_checkpoint(checkpoint=checkpoint, load_last=load_last, force=force, name=name)
	# 	self.load_configs(force=force, clear_info=clear_info)

	# def select_checkpoint(self, checkpoint=None, load_last=True, force=False, name=None):
	#
	# 	note = ('last' if load_last else 'best') if checkpoint is None else checkpoint
	# 	print('Selecting checkpoint: {}'.format(note))
	#
	# 	valid = []
	#
	# 	for run in self.active:
	# 		if 'ckpt_path' not in run or force:
	#
	# 			if name is None:
	#
	# 				if checkpoint is None:
	# 					ckpt_path = find_checkpoint(run.path, load_last=load_last)
	# 				else:
	# 					ckpt_path = os.path.join(run.path, 'checkpoint_{}.pth.tar'.format(checkpoint))
	#
	# 				name_req = 'checkpoint_'
	# 			else:
	# 				name_req = name
	# 				ckpt_path = os.path.join(run.path, name)
	#
	# 			if name_req not in ckpt_path or not os.path.isfile(ckpt_path):
	# 				print('{} has no checkpoint'.format(run.name))
	# 				continue
	#
	# 			run.ckpt_path = ckpt_path
	#
	# 			ckpt_name = os.path.basename(run.ckpt_path)
	#
	# 			if name is None:
	# 				try:
	# 					ckpt_num = int(ckpt_name.split('_')[-1].split('.')[0])
	# 					run.meta.ckpt = ckpt_num
	# 				except ValueError as e:
	# 					print(run.ckpt_path, ckpt_name)
	# 					raise e
	#
	# 		valid.append(run)
	#
	# 	self.set_active(valid)
	#
	# def load_records(self, pbar=None, force=False):
	#
	# 	runs = self.active
	# 	if pbar is not None:
	# 		runs = pbar(runs)
	# 		runs.set_description('Loading records')
	#
	# 	for run in runs:
	# 		if 'records' not in run or force:
	# 			ckpt = torch.load(run.ckpt_path)
	# 			run.records = ckpt['records']
	#
	#
	# def load_configs(self, force=False, clear_info=False, excluded_info=[],
	#                  update_active=True, **kwargs):
	#
	# 	excluded_info = set(excluded_info)
	#
	# 	self.map(lambda r: r.load_config(force=force, clear_info=clear_info, excluded_info=excluded_info), **kwargs)
	#
	# 	if update_active:
	# 		valid = [run for run in self.active if 'config' in run]
	# 		self.set_active(valid)
	# 	print('Loaded configs')

	def set_active(self, active=None):
		if active is None:
			active = self.full_info.copy()
		self.active = active
		for i, run in enumerate(self.active):
			run.idx = i
		self.name2idx = None
		self.output = None
	def extend(self, runs):
		new = []
		for item in runs:
			if item.name not in self:
				new.append(item)
		if len(new):
			self.set_active(self.active + new)
	def append(self, item):
		if item in self: # dont allow duplicates
			return
		self.append(item)
		self.set_active(self.active)

	def add(self, *names):
		runs = []

		for name in names:
			for run in self.full_info:
				if name == run.name:
					runs.append(run)

		if len(runs):
			self.extend(runs)

			print('Added {} run(s)'.format(len(runs)))

	def clear(self):
		self.set_active([])

	def store_selection(self, name, runs=None):
		self.selections[name] = self.active.copy() if runs is None else runs

	def switch_selection(self, name):
		self.set_active(self.selections[name])

	def view_selections(self):
		for k,v in self.selections.items():
			print('{} : holds {} runs'.format(k, len(v)))

	def clear_selections(self):
		self.selections.clear()

	def invert(self):
		inv = [r for r in self.full_info if r.name not in self]
		self.set_active(inv)

	def _update_names(self):
		if self.name2idx is not None:
			return
		self.name2idx = {run.name:i for i,run in enumerate(self.active)}

	def __call__(self, name):
		if isinstance(name, int):
			return self.active[name]
		if isinstance(name, str):
			self._update_names()
			idx = self.name2idx[name]
			return self.active[idx]

		raise Exception('{} not found'.format(name))

	def __contains__(self, item):
		self._update_names()
		return item in self.name2idx

	def __getitem__(self, item):
		if self.active is None:
			print('From full')
			runs = self.full_info
		else:
			runs = self.active
		return runs[item]

	def filter_idx(self, *indicies):
		indicies = set(indicies)
		self.set_active([run for i, run in enumerate(self.active) if i in indicies])

	def sort_by(self, criterion='date', reverse=False):
		if criterion == 'date':
			active = sorted(self.active, key=lambda r: r.meta.date, reverse=reverse)
		elif criterion == 'job':
			active = sorted(self.active, key=lambda r: r.meta.job, reverse=reverse)
		elif criterion == 'model':
			active = sorted(self.active, key=lambda r: r.meta.model, reverse=reverse)
		elif criterion == 'data':
			active = sorted(self.active, key=lambda r: r.meta.dataset, reverse=reverse)
		elif criterion == 'ckpt':
			active = sorted(self.active, key=lambda r: r.meta.ckpt, reverse=reverse)
		else:

			active = criterion(self.active)

			# raise Exception('Unknown criterion: {}'.format(criterion))

		self.set_active(active)

		return self

	def filter(self, criterion):
		active = [run for run in self.active if criterion(run)]
		# print('{} remaining'.format(len(self.active)))
		self.set_active(active)
		return self

	def filter_checkpoints(self, num, cname='checkpoint_{}.pth.tar'):

		cname = cname.format(num)

		remaining = []
		for run in self.active:
			if cname in os.listdir(run.path):
				remaining.append(run)

		self.set_active(remaining)
		return self


	# def filter_complete(self, complete=None):
	# 	remaining = self._check_incomplete(complete=complete, show=False, negate=True)
	# 	self.set_active(remaining)
	# 	return self
	# def filter_incomplete(self, complete=None):
	# 	remaining = self._check_incomplete(complete=complete, show=False, negate=False)
	# 	self.set_active(remaining)
	# 	return self

	def filter_sel(self, sel): # for slices
		self.set_active(self.active[sel])

	def filter_min(self, date=None, job=None, ckpt=None):

		if job is not None:
			self.sort_by('job')
			jobs = sorted([run.meta.job[0] for run in self.active])
			idx = bisect_left(jobs, str(job).zfill(4))
			self.set_active(self.active[idx:])

		if date is not None:
			self.sort_by('date')
			dates = sorted([run.meta.date[0] for run in self.active])
			idx = bisect_left(dates, date)
			self.set_active(self.active[idx:])

		if ckpt is not None:
			self.sort_by('ckpt')
			ckpts = sorted([run.meta.ckpt for run in self.active])
			idx = bisect_left(ckpts, ckpt)
			self.set_active(self.active[idx:])

		return self

	def filter_dates(self, *dates):
		remaining = []
		dates = set(dates)

		for run in self.active:
			day, time = run.meta.date
			if day in dates:
				remaining.append(day)

		self.set_active(remaining)
		return self

	def filter_jobs(self, *jobs):
		remaining = []
		allowed = {str(job) for job in jobs}

		filled = set()
		for job in allowed:
			if '-' in job:
				j,p = job.split('-')
				filled.add('{}-{}'.format(j.zfill(4),p.zfill(2)))
		allowed.update(filled)

		allowed.update({job.zfill(4) for job in allowed})

		for run in self.active:
			job, sch, proc = run.meta.job
			long_job = '{}-{}'.format(job, proc)
			# print(long_job)
			if job in allowed or long_job in allowed:
				remaining.append(run)

		self.set_active(remaining)
		return self

	def filter_strs(self, *strs):
		remaining = []
		for run in self.active:
			for s in strs:
				if ('!' == s[0] and s[1:] not in run.name) or s in run.name:
					remaining.append(run)

		self.set_active(remaining)
		return self

	def filter_models(self, *models):
		remaining = []
		models = set(models)
		for run in self.active:
			model = run.meta.model
			if model in models:
				remaining.append(run)

		self.set_active(remaining)
		return self

	def filter_datasets(self, *datasets):
		remaining = []
		datasets = set(datasets)
		for run in self.active:
			data = run.meta.dataset
			if data in datasets:
				remaining.append(run)

		self.set_active(remaining)
		return self

	def clear_filters(self):
		self.set_active()
		return self

	# def select(self, model=None, dataset=None):
	# 	if model is not None:
	# 		self.filter(lambda r: r.config.info.model_type == model)
	# 	if dataset is not None:
	# 		self.filter(lambda r: r.config.info.dataset_type == dataset)
	# 	return self


	# def options(self, models=True, datasets=True):
	#
	# 	out = []
	#
	# 	if models:
	# 		mopts = set()
	# 		for run in self.active:
	# 			mopts.add(run.config.info.model_type)
	# 		out.append(mopts)
	#
	# 	if datasets:
	# 		dopts = set()
	# 		for run in self.active:
	# 			dopts.add(run.config.info.dataset_type)
	# 		out.append(dopts)
	#
	# 	if len(out) == 1:
	# 		return out[0]
	# 	return out

	def _torun(self, x):
		if x is not None and not isinstance(x, Run):
			x = self(x) if isinstance(x, str) else self[x]
		return x

	def compare(self, base, other=None, bi=False, ignore_keys=None):

		base, other = self._torun(base), self._torun(other)

		if 'config' not in base:
			self.load_configs()

		protected_keys = self._extended_protected_keys.copy() if other is None else self._protected_keys.copy()

		if other is None:
			other = base.config
			base = base.base
		else:
			base, other = base.config, other.config

		if ignore_keys is not None:
			protected_keys.update(ignore_keys)

		return compare_config(base, other=other, bi=bi, ignore_keys=protected_keys)

	# def show_unique(self, ignore_keys=None):
	#
	# 	for i, run in enumerate(self.active):
	#
	# 		print('{:>3}) {}'.format(i, run.name))
	#
	# 		if 'diffs' not in run:
	# 			run.diffs = self.compare(run, ignore_keys=ignore_keys)
	# 		diffs = run.diffs
	# 		base = run.base
	#
	# 		for ks in util.flatten_tree(diffs):
	# 			print('{}{} - {} ({})'.format('\t', '.'.join(map(str, ks)), diffs[ks], (base[ks] if ks in base else '_')))#, util.deep_get(diffs, ks)))
	# 		print()
	#
	# 		run.diffs = diffs

	def _get_unique(self, run, ignore_keys=None):
		if 'diffs' not in run:
			run.diffs = self.compare(run, ignore_keys=ignore_keys)
		diffs = run.diffs
		base = run.base

		unique = ['{} - {} ({})'.format('.'.join(map(str, ks)), diffs[ks],
			                              (base[ks] if ks in base else '_'))
		          for ks in util.flatten_tree(diffs)]

		return unique

	def show(self, *props, ignore_keys=None, indent='\t', manuals=[]):

		props = set(props)

		for i, run in enumerate(self.active):
			print('{:>3}) {}'.format(i, run.name))

			if 'all' in props or 'checkpoint' in props or 'ckpt' in props:
				est = ' (approx {} steps)'.format(run.config.output.save_freq * run.meta.ckpt) if 'config' in run else ''
				print('{}Checkpoint: {}{}'.format(indent, run.meta.ckpt, est))

			if ('all' in props or 'steps' in props) and 'records' in run:
				print('{}Steps: {}'.format(indent, run.records['total_steps']))

			if ('all' in props or 'stats' in props) and 'stats' in run:
				stats = ['{:>10} : {:3.2f}'.format(*item) for item in run.stats.items()]
				print('\n'.join(['{}{}'.format(indent, line) for line in stats]))
				#print()

			if 'all' in props or 'unique' in props:
				unique = self._get_unique(run, ignore_keys=ignore_keys)
				print('\n'.join(['{}{}'.format(indent, line) for line in unique]))

			if len(manuals):
				info = []
				for manual in manuals:
					info.extend(manual(run).split('\n'))
				print('\n'.join(['{}{}'.format(indent, line) for line in info]))

			if len(props) or len(manuals):
				print()


	def start_tb(self, port=None):
		assert self.tbout is not None, 'no tbout'

		if self.tb is None:

			argv = [None, '--logdir', self.tbout]
			if port is not None:
				argv.extend(['--port', str(port)])

			tb = program.TensorBoard()
			tb.configure(argv=argv)
			self.tb_url = tb.launch()
			self.tb = tb
		print('Tensorboard started: {}'.format(self.tb_url))

	def clear_links(self):
		assert self.tbout is not None, 'no tbout'
		for name in os.listdir(self.tbout):
			os.unlink(os.path.join(self.tbout, name))

		for run in self.active:
			for fname in os.listdir(run.path):
				full = os.path.join(run.path, fname)
				if os.path.islink(full):
					os.unlink(full)

	def _val_format(self, val):
		if isinstance(val, (tuple, list)):
			return '{}=' + ','.join(map(str,val))
		if isinstance(val, float):
			return '{}=' + '{:.2g}'.format(val).replace('.', 'p')
		if val == '__removed__':
			return 'no-{}'
		return '{}=' + str(val)

	def link(self, name_fmt='{name}'):

		assert self.tbout is not None, 'no tbout'

		# assert include_dataset or include_date or include_diffs \
		#        or include_idx or include_job or include_model, 'no name for links'

		for i, run in enumerate(self.active):

			unique = None
			if 'unique' in name_fmt:
				if 'diffs' not in run:
					run.diffs = self.compare(run)

				if len(run.diffs):

					unique = '__'.join(self._val_format(run.diffs[ks]).format('.'.join(ks[1:]))
					                 for ks in util.flatten_tree(run.diffs))

				else:
					unique = 'default'

			if 'date' in run.config.info:
				date = run.config.info.date
			elif isinstance(run.config.output._logged_date, str):
				date = run.config.output._logged_date
			else:
				date = run.name.split('_')[-1]

			name = name_fmt.format(idx=i,
			                       unique=unique,
			                       name=run.name,
			                       model=run.config.info.model_type,
			                       dataset=run.config.info.dataset_type,
			                       date=date,
			                       job=run.config.name.split('_')[-1])

			link_path = os.path.join(self.tbout, name)

			if os.path.islink(link_path): # avoid duplicate links
				os.unlink(link_path)
			if 'link' in run and os.path.islink(run.link): # avoid duplicate models
				os.unlink(run.link)

			os.system('ln -s {} {}'.format(run.path, link_path))
			run.link = link_path


# class Run(util.adict):
#
# 	def clear(self):
# 		if 'state' in self:
# 			del self.state
# 			torch.cuda.empty_cache()
#
# 	def reset(self, state=None):
# 		self.clear()
#
# 		if state is None:
# 			state = util.adict()
#
# 		if 'ckpt_path' not in state:
# 			state.ckpt_path = self.ckpt_path
#
# 		self.state = state
# 		return state
#
# 	def load(self, **kwargs):
# 		if 'state' not in self:
# 			self.reset()
#
# 		self._manager._load_fn(self.state, **kwargs)
#
# 	def load_into(self, name, key=None):
# 		if key is None:
# 			key = name.split('.')[0]
# 		if key not in self:
# 			path = os.path.join(self.path, name)
# 			if not os.path.isfile(path):
# 				print('{} not found in: {}'.format(name, self.name))
# 			else:
# 				self[key] = torch.load(path)
#
# 	def load_config(self, force=False, clear_info=False, excluded_info=[]):
# 		if 'config' not in self or force:
#
# 			fname = os.path.join(self.path, 'config.yml')
#
# 			if not os.path.isfile(fname):
# 				print('{} has no config'.format(self.name))
# 				return
#
# 			config = get_config(fname)
#
# 			if clear_info:
# 				del self.config.info
# 			elif 'info' in config:
#
# 				for k, v in config.info.items():
# 					if k in excluded_info:
# 						pass
# 					elif 'dataset_type' == k:
# 						new = v
# 						# if 'dataset' in self.meta:
# 						# 	assert self.meta.dataset == new, '{} vs {}'.format(self.meta.dataset, new)
# 						self.meta.dataset = new
# 					elif 'model_type' == k:
# 						new = v
# 						# if 'model' in self.meta:
# 						# 	assert self.meta.model == new, '{} vs {}'.format(self.meta.model, new)
# 						self.meta.model = new
# 					elif 'history' == k:
# 						self.meta.history = config.info.history
# 					elif 'date' == k:
# 						new = tuple(v.split('-'))
# 						# if 'date' in self.meta:
# 						# 	assert self.meta.date == new, '{} vs {}'.format(self.meta.date, new)
# 						self.meta.date = new
# 					else:
# 						self.meta[k] = v
# 			if 'job' in config:
# 				self.meta.job = config.job.num, config.job.ID, config.job.ps
#
# 			try:
# 				base = get_base_config(config)
# 			except Exception:
# 				# print('WARNING: loading base failed')
# 				base = None
#
# 			if base is None:
# 				print('{} has no base'.format(self.name))
# 				return
#
# 			self.config = config
# 			self.base = base
#
# 	def __str__(self):
# 		return self.name
#
# 	def __repr__(self):
# 		return 'Run({})'.format(self.name)
#
# 	def run(self, **kwargs):
# 		self._manager._run_model_fn(self.state, **kwargs)
#
# 	def evaluate(self, force=False, pbar=None):
#
# 		jobs = self._manager._eval_fns.items()
# 		# if pbar is not None:
# 		# 	jobs = pbar(jobs, total=len(self._manager._eval_fns))
#
# 		if 'evals' not in self.state:
# 			self.state.evals = {}
# 		results = self.state.evals
#
# 		for k, fn in jobs:
# 			if k not in results or force:
# 				print('--- Evaluating: {}'.format(k))
# 				start = time.time()
# 				# if pbar is not None:
# 				# 	jobs.set_description('EVAL: {}'.format(k))
# 				results[k] = fn(self.state, pbar=pbar)
# 				print('... took: {:3.2f}'.format(time.time() - start))
# 			print('{}: {}\n'.format(k, results[k]))
#
# 		# self.state.evals = results
# 		return results
#
# 	def visualize(self, pbar=None):
#
# 		jobs = self._manager._viz_fns.items()
# 		if pbar is not None:
# 			jobs = pbar(jobs, total=len(self._manager._viz_fns))
#
# 		results = {}
# 		for k, fn in jobs:
# 			start = time.time()
# 			if pbar is None:
# 				print('--- Visualizing: {}'.format(k))
# 			else:
# 				jobs.set_description('VIZ: {}'.format(k))
#
# 			results[k] = [Visualization(fig) for fig in fn(self.state, pbar=pbar)]
#
# 			if pbar is None:
# 				print('... took: {:3.2f}\n'.format(time.time() - start))
#
# 		if pbar is not None:
# 			jobs.close()
#
# 		self.state.figs = results
# 		return results
#
# 	def save(self, save_dir=None, fmtdir='{}', overwrite=False, append_ckpt=True,
# 	         include_checkpoint=True, include_config=True, include_original=True, include_results=True,
# 	         img_ext='png', vid_ext='mp4'):
#
# 		if save_dir is None:
# 			save_dir = self._manager.save_dir
#
# 		name = self.name
# 		if append_ckpt and 'ckpt' in self.meta and 'ckpt' not in name:
# 			num = self.meta.ckpt
# 			name = '{}_ckpt{}'.format(name, num)
#
# 		save_path = os.path.join(save_dir, fmtdir.format(name))
#
# 		print('Saving results to: {}'.format(save_path))
#
# 		try:
# 			if overwrite:
# 				util.create_dir(save_path)
# 			else:
# 				os.makedirs(save_path)
# 		except FileExistsError:
# 			print('ERROR: File already exists, you can overwrite using the "overwrite" arg')
# 		else:
# 			if include_checkpoint:
# 				src = self.ckpt_path
# 				dest = os.path.join(save_path, 'model.pth.tar')
# 				shutil.copyfile(src, dest)
# 				print('\tModel saved')
# 			if include_config:
# 				src = os.path.join(self.path, 'config.yml')
# 				dest = os.path.join(save_path, 'config.yml')
# 				shutil.copyfile(src, dest)
# 				print('\tConfig saved')
# 			if include_original:
# 				with open(os.path.join(save_path, 'original_ckpt_path.txt'), 'w') as f:
# 					f.write(self.ckpt_path)
#
# 			if 'state' in self:
# 				if 'figs' in self.state:
# 					for name, figs in self.state.figs.items():
# 						if len(figs) == 1:
# 							path = os.path.join(save_path, name)
# 							figs[0].save(path, img_ext=img_ext, vid_ext=vid_ext)
# 						else:
# 							for i, fig in enumerate(figs):
# 								path = os.path.join(save_path, '{}{}'.format(name, str(i).zfill(3)))
# 								fig.save(path, img_ext=img_ext, vid_ext=vid_ext)
#
# 					print('\tVisualization saved')
#
# 				if 'evals' in self.state:
# 					with open(os.path.join(save_path, 'eval.txt'), 'w') as f:
# 						for k, v in self.state.evals.items():
# 							f.write('{} : {}\n'.format(k, str(v)))
# 					torch.save(self.state.evals, os.path.join(save_path, 'eval.pth.tar'))
# 					print('\tEvaluation saved')
#
# 				if include_results and 'results' in self.state:
# 					torch.save(self.state.results, os.path.join(save_path, 'results.pth.tar'))
# 					print('\tResults saved: {}'.format(', '.join(map(str, self.state.results.keys()))))
#
# 		return save_path

#
# def compare_config(base, other=None, bi=False, ignore_keys=None):
# 	# vs_default = False
#
# 	protected_keys = {'name', 'info', 'job', 'save_dir'}
# 	if other is None:
#
# 		protected_keys.update({
# 		'_logged_date', 'din', 'dout', 'info', 'dataroot', 'saveroot', 'run_mode', 'dins', 'douts',
# 		})
#
# 		if 'history' not in base.info:
# 			raise Exception('Unable to load default - no history found')
#
# 		other = base
# 		base = get_base_config(base)
#
# 	if ignore_keys is not None:
# 		protected_keys.update(ignore_keys)
#
# 	diffs = ConfigDict()#util.NS()
# 	_compare_configs(base, other, diffs=diffs, protected_keys=protected_keys)
#
# 	if bi:
# 		adiffs = ConfigDict()#util.NS()
# 		_compare_configs(other, base, diffs=adiffs, protected_keys=protected_keys)
# 		return diffs, adiffs
#
# 	return diffs
#
# def get_base_config(base):
# 	if 'info' in base and 'history' in base.info:
# 		return parse_config(base.info.history)
#
# def _compare_configs(base, other, diffs, protected_keys=None):
#
# 	for k,v in base.items():
#
# 		if k in protected_keys:
# 			pass
# 		elif k not in other:
# 			diffs[k] = '__removed__'
# 		elif isinstance(v, ConfigDict):
# 			_compare_configs(v, other[k], diffs=diffs[k], protected_keys=protected_keys)
# 			if len(diffs[k]) == 0:
# 				del diffs[k]
# 		elif v != other[k]:
# 			diffs[k] = other[k]
#
# 	for k,v in other.items():
# 		if k not in protected_keys and k not in base:
# 			diffs[k] = v
#
#
#
# def render_format(raw, unfolded=False):
# 	unfolded = True
# 	if isinstance(raw, set):
# 		itr = dict()
# 		for i, el in enumerate(raw):
# 			itr['s{}'.format(i)] = render_format(el, unfolded)
# 		return itr
# 	elif isinstance(raw, dict):
# 		return dict((str(k), render_format(v, unfolded)) for k, v in raw.items())
# 	elif isinstance(raw, list):
# 		return list(render_format(el) for el in raw)
# 		itr = dict()
# 		for i, el in enumerate(raw):
# 			itr['l{}'.format(i)] = render_format(el, unfolded)
# 		return itr
# 	elif isinstance(raw, tuple):
# 		return list(render_format(el) for el in raw)
# 		itr = dict()
# 		for i, el in enumerate(raw):
# 			itr['t{}'.format(i)] = render_format(el, unfolded)
# 		return itr
# 	return str(raw)
#
#
# import uuid
# from IPython.display import display_javascript, display_html
#
#
# class render_dict(object):
# 	def __init__(self, json_data):
# 		self.json_str = render_format(json_data)
#
# 		# if isinstance(json_data, dict):
# 		#     self.json_str = json_data
# 		#     #self.json_str = json.dumps(json_data)
# 		# else:
# 		#     self.json_str = json
# 		self.uuid = str(uuid.uuid4())
#
# 	def _ipython_display_(self):
# 		display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid),
# 		             raw=True
# 		             )
# 		display_javascript("""
# 		require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
# 		  renderjson.set_show_to_level(1)
# 		  document.getElementById('%s').appendChild(renderjson(%s))
# 		});
# 		""" % (self.uuid, self.json_str), raw=True)






