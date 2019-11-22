import sys, os
import torch
import torchvision
from torchvision.datasets import MNIST
import numpy as np

import random

from .. import framework as fm
from .. import util
from . import general as gen

class MNIST_Walker(fm.Env):
	def __init__(self, batch_size=1, traindata=True, download=False, stochastic=False,
	             sparse_reward=True, random_goal=False, loop=True):
		spec = gen.EnvSpec(obs_space=gen.Continuous_Space(shape=(batch_size ,1 ,28 ,28) ,),
		                   act_space=gen.Discrete_Space(choices=3, shape=(batch_size,)),
		                   horizon=20)
		super(MNIST_Walker, self).__init__(spec, ID='MNIST-Walker')

		self.batch_size = batch_size
		self.gen = np.random.RandomState()
		self.sparse_reward = sparse_reward
		self.random_goal = random_goal
		self.loop = loop

		self._act_deltas = torch.LongTensor([-1, 0, 1])

		assert not stochastic, 'not set up'

		data = MNIST(os.path.join(util.FOUNDATION_DIR, '..', 'data', 'mnist'),
		             train=traindata, download=download, transform=torchvision.transforms.ToTensor())

		self._lbls = data.train_labels if traindata else data.test_labels
		self._imgs = data.train_data if traindata else data.test_data
		self._imgs = self._imgs.unsqueeze(1)

		full = torch.arange(len(self._lbls)).long()
		self._idx = [full[self._lbls == i].long().numpy() for i in range(10)]

	def seed(self, seed=None):
		if seed is None:
			seed = np.random.randint(2**31)

		self.gen = np.random.RandomState(seed)

		return seed

	def _sample(self, digits):
		return torch.LongTensor([self.gen.choice(self._idx[digit]) for digit in digits])

	def _get_obs(self, lbls):
		self._idx_state = self._sample(lbls)
		imgs = self._imgs[self._idx_state]

		if self.batch_size == 1:
			return imgs.squeeze()
		return imgs

	def reset(self, init_state=None):
		self.goal = torch.from_numpy(self.gen.randint(10, size=self.batch_size)) if self.random_goal else torch.zeros(self.batch_size).long()

		self.lbl_state = torch.from_numpy(self.gen.randint(0,10,size=self.batch_size))
		obs = self._get_obs(self.lbl_state)
		return obs

	def step(self, action): # returns: next_obs, reward, done, info

		self.lbl_state += self._act_deltas[action]
		if self.loop:
			self.lbl_state %= 10
		else:
			self.lbl_state = self.lbl_state.clamp(0,9)

		reward = self.lbl_state - self.goal
		if self.sparse_reward:
			reward = (reward == 0)
		elif self.loop:
			reward = reward.abs()
			reward[reward > 5] = 10 - reward[reward > 5]
			reward = -reward
		else:
			reward = -reward.abs()
		reward = reward.float()

		return self._get_obs(self.lbl_state), reward, False, {}

	def render(self, *args, **kwargs):
		return self._imgs[self._idx_state]

