
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


