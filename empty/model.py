
import sys, os, time, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import configargparse

import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt

import foundation as fd
from foundation import models
from foundation import util
from foundation import train
from foundation import data



# TODO: define models here



def get_options():
	parser = train.get_parser()

	# TODO: add any args custom args here

	return parser

def get_data(args):

	# TODO: build datasets here

	raise NotImplementedError

def get_model(args):

	# TODO: build model/optim here

	raise NotImplementedError




if __name__ == '__main__':

	argv = None

	try:
		train.run_full(get_options, get_data, get_model, argv=argv)

	except KeyboardInterrupt:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()

	except:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()
		ipdb.post_mortem(tb)




