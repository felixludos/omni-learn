
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

MY_PATH = os.path.dirname(os.path.abspath(__file__))

train.register_config('model', os.path.join(MY_PATH, 'config', 'test.yaml'))
train.register_config('pycharm', os.path.join(MY_PATH, 'config', 'pycharm.yaml'))

class Model(fd.Visualizable, fd.Trainable_Model):
	def __init__(self, info):

		net = info.pull('net')
		criterion = info.pull('criterion', 'cross-entropy')
		optim_info = info.pull('optim', None)

		super().__init__(net.din, net.dout)

		self.net = net
		self.criterion = util.get_loss_type(criterion)

		self.stats.new('accuracy', 'confidence')

		if optim_info is not None:
			self.set_optim(optim_info)


	def forward(self, x):
		return self.net(x)

	def _visualize(self, info, logger):
		# if self._viz_counter % 5 == 0:
		# 	pass

		conf, pick = info.pred.max(-1)

		confidence = conf.detach()
		correct = pick.sub(info.y).eq(0).float().detach()

		self.stats.update('confidence', confidence.mean())
		self.stats.update('accuracy', correct.mean())


	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		# compute loss
		x, y = batch

		out.x, out.y = x, y

		pred = self(x)
		out.pred = pred

		loss = self.criterion(pred, y)
		out.loss = loss

		if self.train_me():
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out

train.register_model('model', Model)

def get_data(A, mode='train'):
	return train.default_load_data(A, mode=mode)

def get_model(A):
	return train.default_create_model(A)

def get_name(A):
	assert 'name' in A, 'Must provide a name manually'
	return A.name


if __name__ == '__main__':
	sys.exit(train.main(get_data=get_data, get_model=get_model, get_name=get_name))




