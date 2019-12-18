
import sys, os, shutil
import numpy as np
import torch

from torch import multiprocessing as mp

from .. import util
from .config import get_config, Config, parse_config
from .loading import find_checkpoint

# import tensorflow as tf
# from tensorboard import main as tb
from tensorboard import program



class Run(util.tdict):
	pass

class Run_Manager(object):

	def __init__(self, path=None, recursive=False, tbout=None):

		if path is None:
			assert 'FOUNDATION_SAVE_DIR' in os.environ, 'no path provided, and no default save dir set'
			path = os.environ['FOUNDATION_SAVE_DIR']

		self.master_path = path
		self.recursive = recursive

		self._protected_keys = {'name', 'info', 'job',
		                        'save_dir'}
		self._extended_protected_keys = {
			'_logged_date', 'din', 'dout', 'info', 'dataroot', 'saveroot', 'run_mode', 'dins', 'douts',
		}

		self.tbout = tbout
		if self.tbout is not None:
			util.create_dir(self.tbout)
			print('{} is available to view runs on tensorboard'.format(self.tbout))
		self.tb = None

		self.refresh()

	def _collect_runs(self, path=None, load_last=True, clear_info=False):
		rootdir = self.master_path if path is None else os.path.join(self.master_path, path)
		for name in sorted(os.listdir(rootdir)):

			run_name = name if path is None else os.path.join(path, name)
			run_path = os.path.join(rootdir, name) if path is None else os.path.join(self.master_path, name)
			if os.path.isdir(run_path):
				if 'config.yml' in os.listdir(run_path):

					try:
						ckpt_path = find_checkpoint(run_path, load_last=load_last)
						run = Run(name=run_name, path=run_path)
						run.ckpt_path = ckpt_path
						run.config = get_config(os.path.join(run_path, 'config.yml'))

						if clear_info:
							del run.config.info

						if 'model_type' not in run.config.info:
							run.config.info.model_type = run.config.model._type
						if 'dataset_type' not in run.config.info:
							run.config.info.dataset_type = run.config.dataset.name

					except FileNotFoundError:
						pass
					else:
						self.full_info.append(run)
				elif self.recursive:
					self._collect_runs(run_name, load_last=load_last, clear_info=clear_info)

	def refresh(self, load_last=True, clear_info=False): # collects info from all runs

		self.full_info = []

		self._collect_runs(load_last=load_last, clear_info=clear_info)

		self.run2idx = {r.name: i for i,r in enumerate(self.full_info)}

		print('Found {} runs'.format(len(self.full_info)))

		self.active = None
		self.clear_filters()

	def show(self):
		for i,r in enumerate(self.active):
			print('{:>3} - {}'.format(i, r.name))

	def getrun(self, name):
		return self.full_info[self.run2idx[name]]

	def __getitem__(self, item):
		if self.active is None:
			print('From full')
			return self.full_info[item]
		return self.active[item]

	def filter(self, criterion):

		self.active = [run for run in self.active if criterion(run)]
		# print('{} remaining'.format(len(self.active)))

		return self

	def clear_filters(self):
		self.active = self.full_info.copy()
		return self

	def select(self, model=None, dataset=None):

		if model is not None:
			self.filter(lambda r: r.config.info.model_type == model)
		if dataset is not None:
			self.filter(lambda r: r.config.info.dataset_type == dataset)
		return self

	def options(self, models=True, datasets=True):

		out = []

		if models:
			mopts = set()
			for run in self.active:
				mopts.add(run.config.info.model_type)
			out.append(mopts)

		if datasets:
			dopts = set()
			for run in self.active:
				dopts.add(run.config.info.dataset_type)
			out.append(dopts)

		if len(out) == 1:
			return out[0]
		return out

	def _torun(self, x):
		if not isinstance(x, Run):
			x = self.getrun(x) if isinstance(x, str) else self[x]
		return x

	def _compare_configs(self, base, other, diffs, protected_keys=None):
		if protected_keys is None:
			protected_keys = self._protected_keys

		for k,v in base.items():

			if k in protected_keys:
				pass
			elif k not in other:
				diffs[k] = '__removed__'
			elif isinstance(v, Config):
				self._compare_configs(v, other[k], diffs=diffs[k], protected_keys=protected_keys)
				if len(diffs[k]) == 0:
					del diffs[k]
			elif v != other[k]:
				diffs[k] = other[k]

		for k,v in other.items():
			if k not in protected_keys and k not in base:
				diffs[k] = v

	def compare(self, base, other=None, bi=False, ignore_keys=None):
		# vs_default = False
		base_run = self._torun(base)
		base = self._torun(base).config
		protected_keys = self._protected_keys.copy()
		if ignore_keys is not None:
			protected_keys.update(ignore_keys)
		if other is None:
			if 'history' not in base.info:
				raise Exception('Unable to load default - no history found')

			other = base
			base = parse_config(base.info.history)
			base_run.base = base
			# vs_default = True
			protected_keys.update(self._extended_protected_keys)

		else:
			other = self._torun(other).config

		diffs = Config()#util.NS()
		self._compare_configs(base, other, diffs=diffs, protected_keys=protected_keys)

		if bi:
			adiffs = Config()#util.NS()
			self._compare_configs(other, base, diffs=adiffs, protected_keys=protected_keys)
			return diffs, adiffs

		return diffs


	def show_unique(self, ignore_keys=None):

		for i, run in enumerate(self.active):

			print('{}) {}'.format(i, run.name))

			diffs = self.compare(run, ignore_keys=ignore_keys)
			base = run.base

			for ks in util.flatten_tree(diffs):
				print('{}{} - {} ({})'.format('\t', '.'.join(map(str, ks)), diffs[ks], (base[ks] if ks in base else '_')))#, util.deep_get(diffs, ks)))
			print()

			run.diffs = diffs

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
			                       date=date)

			link_path = os.path.join(self.tbout, name)

			if os.path.islink(link_path): # avoid duplicate links
				os.unlink(link_path)
			if 'link' in run and os.path.islink(run.link): # avoid duplicate models
				os.unlink(run.link)

			os.system('ln -s {} {}'.format(run.path, link_path))
			run.link = link_path








def render_format(raw, unfolded=False):
	unfolded = True
	if isinstance(raw, set):
		itr = dict()
		for i, el in enumerate(raw):
			itr['s{}'.format(i)] = render_format(el, unfolded)
		return itr
	elif isinstance(raw, dict):
		return dict((str(k), render_format(v, unfolded)) for k, v in raw.items())
	elif isinstance(raw, list):
		return list(render_format(el) for el in raw)
		itr = dict()
		for i, el in enumerate(raw):
			itr['l{}'.format(i)] = render_format(el, unfolded)
		return itr
	elif isinstance(raw, tuple):
		return list(render_format(el) for el in raw)
		itr = dict()
		for i, el in enumerate(raw):
			itr['t{}'.format(i)] = render_format(el, unfolded)
		return itr
	return str(raw)


import uuid
from IPython.display import display_javascript, display_html


class render_dict(object):
	def __init__(self, json_data):
		self.json_str = render_format(json_data)

		# if isinstance(json_data, dict):
		#     self.json_str = json_data
		#     #self.json_str = json.dumps(json_data)
		# else:
		#     self.json_str = json
		self.uuid = str(uuid.uuid4())

	def _ipython_display_(self):
		display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid),
		             raw=True
		             )
		display_javascript("""
		require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
		  renderjson.set_show_to_level(1)
		  document.getElementById('%s').appendChild(renderjson(%s))
		});
		""" % (self.uuid, self.json_str), raw=True)






