
import sys, os
import numpy as np
import torch

from .. import util
from .config import get_config, Config
from .loading import find_checkpoint


class Run(util.tdict):
	pass

class Run_Manager(object):

	def __init__(self, path=None, recursive=False):

		if path is None:
			assert 'FOUNDATION_SAVE_DIR' in os.environ, 'no path provided, and no default save dir set'
			path = os.environ['FOUNDATION_SAVE_DIR']

		self.master_path = path
		self.recursive = recursive

		self._protected_keys = {'name', 'save_dir'}

		self.refresh()

	def _collect_runs(self, path=None, load_last=True):
		rootdir = self.master_path if path is None else os.path.join(self.master_path, path)
		for name in os.listdir(rootdir):

			run_name = name if path is None else os.path.join(path, name)
			run_path = os.path.join(rootdir, name) if path is None else os.path.join(self.master_path, name)
			if os.path.isdir(run_path):
				if 'config.yml' in os.listdir(run_path):

					try:
						ckpt_path = find_checkpoint(run_path, load_last=load_last)
						run = Run(name=run_name, path=run_path)
						run.ckpt_path = ckpt_path
						run.config = get_config(os.path.join(run_path, 'config.yml'))
					except FileNotFoundError:
						pass
					else:
						self.full_info.append(run)
				elif self.recursive:
					self._collect_runs(run_name, load_last=load_last)

	def refresh(self, load_last=True): # collects info from all runs

		self.full_info = []

		self._collect_runs(load_last=load_last)

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

	# def _equate(self, A, B, diffs=None):
	# 	if isinstance(A, Config):
	# 		return self._compare_configs(A, B, diffs=diffs)
	# 	return A == B

	def _compare_configs(self, base, other, diffs):

		for k,v in base.items():

			if k in self._protected_keys:
				pass
			elif k not in other:
				diffs[k] = '__removed__'
			elif isinstance(v, Config):
				self._compare_configs(v, other[k], diffs=diffs[k])
				if len(diffs[k]) == 0:
					del diffs[k]
			elif v != other[k]:
				diffs[k] = other[k]

		for k,v in other.items():
			if k not in self._protected_keys and k not in base:
				diffs[k] = v

	def compare(self, base, other, bi=False):
		if not isinstance(base, Run):
			base = self.getrun(base) if isinstance(base, str) else self[base]
		if not isinstance(other, Run):
			other = self.getrun(other) if isinstance(other, str) else self[other]

		diffs = util.NS()
		self._compare_configs(base.config, other.config, diffs=diffs)

		if bi:
			adiffs = util.NS()
			self._compare_configs(other.config, base.config, diffs=adiffs)
			return diffs, adiffs

		return diffs






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






