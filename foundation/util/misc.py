
import sys, os
import torch
import numpy as np
#from itertools import imap
from collections import deque, Mapping

from humpack import tdict, tlist, tset

import torch.nn as nn
import torch.multiprocessing as mp
# from torch.utils.data.dataloader import ExceptionWrapper
import random

FD_PATH = os.path.dirname(os.path.dirname(__file__))

primitives = (str, int, float, bool)

class NS(tdict): # NOTE: avoid hasattr! - always returns true (creating new attrs), use __contains__ instead
	'''
	Namespace - like a dictionary but where keys can be accessed as attributes, and if not found will create new NS
	allowing:

	a = NS()
	a.b.c.d = 'hello'
	print(repr(a)) # --> NS('b':NS('c':NS('d':'hello')))

	'''

	def __getitem__(self, key):
		try:
			v = super().__getitem__(key)
			# print(key,v)
			return v
		except KeyError:
			try:
				return super().__getattribute__(key)
			except AttributeError:
				print('**WARNING: defaulting {}'.format(key))
				self.__setitem__(key, self.__class__())
				return super().__getitem__(key)

	def todict(self):
		d = {}
		for k,v in self.items():
			if isinstance(v, NS):
				v = v.todict()
			d[k] = v
		return d

	def __repr__(self):
		return 'NS({})'.format(', '.join(['{}:{}'.format(repr(k), repr(v)) for k,v in self.items()]))



class Simple_Child(object): # a simple wrapper that delegates __getattr__s to some parent attribute

	def __init__(self, *args, _parent=None, **kwargs):
		super().__init__(*args, **kwargs)
		self._parent = _parent

	def __getattribute__(self, item):
		parent = super().__getattribute__('_parent')
		if parent is None:
			return super().__getattribute__(item)
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return parent.__getattribute__(item)


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


