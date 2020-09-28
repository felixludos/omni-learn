
import sys, os, time
import traceback
import math
import torch
import numpy as np
#from itertools import imap
from collections import deque, Mapping

from omnibelt import Simple_Child, Proper_Child, get_now, create_dir, save_yaml, load_yaml, Registry

import omnifig as fig

from humpack import adict, tlist, tset, Table, TreeSpace

import torch.nn as nn
import torch.multiprocessing as mp
# from torch.utils.data.dataloader import ExceptionWrapper
import random

from .farming import make_ghost

FD_PATH = os.path.dirname(os.path.dirname(__file__))

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(FD_PATH),'local_data')
DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(FD_PATH),'trained_nets')


@fig.Component('pull')
def _config_pull(A):
	terms = A.pull('_terms', '<>_term', silent=True)
	args = A.pull('_args', {}, silent=True)
	return A.pull(*terms, **args)


def _process_single(term, A):
	op, val = None, term

	if isinstance(term, (tuple, list)):
		assert len(term) == 2, f'unknown term: {term}'
		op, val = term

	if isinstance(val, (tuple, list)):
		val = A.pull(*val, silent=True)
	elif isinstance(val, str):
		val = A.pull(val, silent=True)
	if op is None or op in {0, 1, 'id', 'i', 'x', ''}:
		return val

	if op in {'minv', '/', 'div'}:
		return 1 / val
	elif op in {'ainv', '-', 'sub'}:
		return -val
	raise Exception(f'unknown op: {op}')

@fig.Component('expr')
def _config_expression(A):  # TODO: boolean ops

	red = A.pull('_reduce', '+', silent=True)

	terms = A.pull('_terms', '<>_term', silent=True)

	if not isinstance(terms, (list, tuple)):
		terms = terms,

	vals = [_process_single(term, A) for term in terms]

	if red in {'+', 'add', 'sum'}:
		out = sum(vals)
	elif red in {'avg', 'average', 'mean'}:
		out = sum(vals) / len(vals)
	elif red in {'*', 'product', 'mul'}:
		out = math.prod(vals)
	elif red in {'%', 'mod'}:
		assert len(vals) == 2, f'bad red: {vals}'
		out = vals[0] % vals[1]
	elif red in {'//', 'idiv'}:
		assert len(vals) == 2, f'bad red: {vals}'
		out = vals[0] // vals[1]
	else:
		raise Exception(f'unknown reduction {red}')

	caste = A.pull('_caste', None, silent=True)

	if caste is None:
		return out
	elif caste == 'int':
		return int(out)
	elif caste == 'str':
		return str(out)
	elif caste == 'float':
		return float(out)
	raise Exception(f'unkonwn caste: {caste}')


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


