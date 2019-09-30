
import sys, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal

from .. import framework as fm
from .. import util

####################
# Mixins
####################

class Scalable_Baseline(fm.Baseline): # Scales the values so that they are within [-1,1]
	def __init__(self, scale_max=False, **other):
		super().__init__(**other)
		self.scale = 1. if scale_max else None

	def learn(self, states, values):
		if self.scale is not None:
			mx = values.abs().max().item()
			self.scale = max(self.scale, mx)
			values = values / self.scale
		super().learn(states, values)

	def forward(self, x):
		if self.scale is None:
			return super().forward(x)
		return self.scale * super().forward(x)


####################
# Baselines
####################


class Linear_Baseline(Scalable_Baseline):

	def __init__(self, **other):
		super().__init__(**other)

		self.model = nn.Linear(self.state_dim, self.value_dim)
		for p in self.model.parameters():
			p.requires_grad = False

		self.stats.new('error-before', 'error-after')

	def get_loss(self, x, y):
		return F.mse_loss(self(x),y).detach()

	def _update(self, states, values):
		states = states.view(-1, self.din)
		values = values.view(-1, self.dout)

		self.stats.update('error-before', self.get_loss(states, values))

		util.solve(states, values, out=self.model)

		self.stats.update('error-after', self.get_loss(states, values))

	def _get_value(self, x):
		return self.model(x)

class Deep_Baseline(Scalable_Baseline):

	def __init__(self, model, batch_size=64, epochs_per_step=10, **other):
		super().__init__(state_dim=model.din, value_dim=model.dout, stats=model.stats, **other)

		self.batch_size = batch_size
		self.epochs = epochs_per_step

		self.model = model
		self.stats.new('error-before', 'error-after')

	def _update(self, states, values):

		self.stats.update('error-before', self.model.get_loss(states, values))

		dataloader = DataLoader(TensorDataset(states, values),
		                        batch_size=self.batch_size, shuffle=True, num_workers=0)

		for epoch in range(self.epochs):
			self.model.train_epoch(dataloader)

		self.stats.update('error-after', self.model.get_loss(states, values))

	def _get_value(self, x):
		return self.model(x)

