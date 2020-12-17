
import sys, os, time
import traceback
from tqdm import tqdm, tqdm_notebook
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
		ptype = A.pull('pbar-type', 'cmd')
		
		self._pbar_cls = tqdm_notebook if ptype in {'notebook', 'jupyter'} else tqdm
		self.pbar = None
		
		self.limit = A.pull('limit', None)
	
	def init_pbar(self, limit=None, **kwargs):
		self.reset()
		if self._pbar_cls is not None:
			if limit is None:
				limit = self.limit
			self.pbar = self._pbar_cls(total=limit, **kwargs)
	
	def update(self, desc=None, n=1):
		if self.pbar is not None:
			self.pbar.update(desc=desc, n=n)
	
	def __call__(self, itr, **kwargs):
		self.reset()
		self.pbar = self._pbar_cls(itr)
		return self.pbar
	
	def __iter__(self):
		self.init_pbar()
		return self
	
	def __next__(self):
		return self.update()
	
	def reset(self):
		if self.pbar is not None:
			self.pbar.close()
	
	def set_description(self, desc):
		if self.pbar is not None:
			self.pbar.set_description(desc)


