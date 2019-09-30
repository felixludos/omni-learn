
import sys, os
import torch
import numpy as np
#from itertools import imap
from collections import deque, Mapping
import torch.nn as nn
import torch.multiprocessing as mp
# from torch.utils.data.dataloader import ExceptionWrapper
import random

class NS(object):
	'''
	Namespace - like a dictionary but where keys can be accessed as attributes
	'''
	def __init__(self, **kwargs):
		self.__dict__ = kwargs
	def __setitem__(self, key, value):
		self.__dict__[key] = value
	def __getitem__(self, key):
		return self.__dict__[key]
	def __len__(self):
		return len(self.__dict__)
	def __iter__(self):
		return self.__dict__.__iter__()
	def __contains__(self, item):
		return self.__dict__.__contains__(item)
	def keys(self):
		return self.__dict__.keys()
	def values(self):
		return self.__dict__.values()
	def items(self):
		return self.__dict__.items()
	def update(self, other):
		if isinstance(other, NS):
			other = other.__dict__
		return self.__dict__.update(other)
	def __repr__(self):
		return 'NameSpace(keys=[' + ', '.join([str(k) for k in self.keys()]) + '])'

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


