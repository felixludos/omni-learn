import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import torch.multiprocessing as mp
import foundation as fd
from foundation import models
from foundation import util
import gym


class NormalizedMLP(fd.Supervised_Model):
	def __init__(self, input_dim, output_dim, norm=True, **args):
		super().__init__(in_dim=input_dim, out_dim=output_dim, criterion=util.get_loss_type('mse'))
		self.norm = models.RunningNormalization(input_dim) if norm else None
		self.net = models.make_MLP(input_dim, output_dim, **args)

	def forward(self, x):
		if self.norm is not None:
			x = self.norm(x)
		return self.net(x)




