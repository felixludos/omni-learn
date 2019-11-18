
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

class NS(tdict): # NOTE: avoid hasattr! - always returns true (creating new attrs), use __contains__ instead
	'''
	Namespace - like a dictionary but where keys can be accessed as attributes
	'''

	def __getitem__(self, key):
		try:
			return super().__getitem__(key)
		except KeyError:
			self.__setitem__(key, NS())
			return super().__getitem__(key)



class Simple_Child(object): # a simple wrapper that delegates __getattr__s to some parent attribute

	def __init__(self, *args, _parent='target', **kwargs):
		super().__init__(*args, **kwargs)
		self._parent = _parent

	def __getattr__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return self.__getattr__(self._parent).__getattribute__(item)


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


