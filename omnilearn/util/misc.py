
import sys, os, time
import traceback
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import math
import torch
import numpy as np
#from itertools import imap
from collections import deque, Mapping

from omnibelt import Simple_Child, Proper_Child, get_now, create_dir, save_yaml, load_yaml, Registry, Singleton

import omnifig as fig

from humpack import adict, tlist, tset, Table, TreeSpace

import torch.nn as nn
import torch.multiprocessing as mp
# from torch.utils.data.dataloader import ExceptionWrapper
import random

from .farming import make_ghost

FD_PATH = os.path.dirname(os.path.dirname(__file__))

# fig.register_config_dir(os.path.join(os.path.dirname(FD_PATH), 'config'))
fig.register_config('origin', os.path.join(os.path.dirname(FD_PATH), 'config', 'origin.yaml'))

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(FD_PATH),'local_data')
DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(FD_PATH),'trained_nets')


def to_np(tensor):
	return tensor.detach().cpu().numpy()

def split_vals(a, groups):
	idx = np.random.choice(np.arange(len(groups)), size=len(a), replace=True, p=groups)
	return [a[g==idx] for g in range(len(groups))]

def count_parameters(model):
	return sum(p.numel() for p in model.parameters())

def create_param(*sizes, requires_grad=True):
	t = torch.empty(*sizes)
	nn.init.xavier_normal_(t)
	return nn.Parameter(t, requires_grad=requires_grad)

@fig.Component('progress-bar')
class Progress_Bar(Singleton):
	def __init__(self, A):
		ptype = A.pull('display-on', 'cmd')
		
		self._pbar_cls = tqdm_notebook if ptype in {'notebook', 'jupyter'} else tqdm
		self.pbar = None
		self.pbariter = None
		
		self.default_limit = A.pull('limit', None)
		
		self.pause_state = None
		self._val = None
		self._desc = None
		self._limit = None
	
	def pause(self):
		self.pause_state = {'limit':self._limit, 'initial':self._val, 'desc':self._desc}
		self.reset()
	
	def unpause(self):
		if self.pause_state is not None:
			self.reset()
			self.init_pbar(**self.pause_state)
			self.pause_state = None
	
	def init_pbar(self, itr=None, limit=None, initial=0, desc=None, total=None, **kwargs):
		self.reset()
		if self._pbar_cls is not None:
			if itr is not None:
				try:
					limit = len(itr)
				except TypeError:
					pass
			if limit is None:
				limit = self.default_limit
			if total is None:
				total = limit
			self._limit = total
			self._val = initial
			self._desc = desc
			self.pbar = self._pbar_cls(itr, total=total, initial=initial, desc=desc, **kwargs)
			# if itr is not None:
			# 	self.pbariter = iter(self.pbar)
	
	def update(self, n=1, desc=None):
		if self.pbar is not None:
			self._desc = desc
			self._val += n
			if desc is not None:
				self.set_description(desc)
			if self.pbariter is not None:
				return next(self.pbariter)
			else:
				self.pbar.update(n=n)
	
	def __call__(self, itr=None, **kwargs):
		self.init_pbar(itr=itr, **kwargs)
		return self
	
	def __iter__(self):
		if self.pbariter is None and self.pbar is not None:
			self.pbariter = iter(self.pbar)
		return self
	
	def __next__(self):
		return self.update()
	
	def close(self):
		self.reset()
	
	def reset(self):
		if self.pbar is not None:
			try:
				self.pbar.close()
			except TypeError:
				pass
			print('\r', end='')
		self.pbar = None
		self.pbariter = None
	
	def set_description(self, desc):
		if self.pbar is not None:
			self.pbar.set_description(desc)


