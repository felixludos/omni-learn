
import sys, os
import numpy as np
import gym
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import copy

import models
import util

class Agent:
	def __init__(self, model, args):
		self.mode = 'train' # train vs eval - mostly for noisy actions
		self.model = model

		self.def_type = args.def_type
		self.cuda = args.cuda

	def train(self): # switch to training mode
		self.mode = 'train'
		self.model.train()

	def eval(self): # switch to eval mode
		self.mode = 'eval'
		self.model.eval()

	def load(self, path):
		pass # should return number of episodes

	def save(self, episodes, path):
		pass # save model to path and save number of episodes

	def snapshot(self):
		'''
		when called, this should return any data to be stored for viz as a dict for which the keys should not change between calls
		'''
		return {}

	def learn(self, state, action, reward, next_state, done):
		'''
		when called, the agent learn using this SARS transition
		'''
		pass

	def get_action(self, state, force_noisy=None): # should be overwritten by agent
		'''
		when in training mode this returns a noisy action, but noise is overruled with 'force_noisy'
		'''
		pass

class NPG(Agent):
	def __init__(self, args, **kwargs):
		super(NPG,self).__init__(args=args, **kwargs)

class DDPG(Agent):
	def __init__(self, args, **kwargs):
		super(DDPG,self).__init__(args=args, **kwargs)

		assert args.policy == 'separate'

		self.target = copy.deepcopy(self.model)

		if self.cuda:
			self.model.cuda()
			self.target.cuda()
		
		self.memory = util.ReplayBuffer(args.buffer_size)
		self.noise = util.OUNoise(args.action_dim)

		self.criterion = nn.MSELoss()

		self.actor_optimizer = models.get_optimizer(args.optimizer, self.model.actor.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		self.critic_optimizer = models.get_optimizer(args.optimizer, self.model.critic.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

		self.discount = args.discount
		self.tau = args.tau
		self.batch_size = args.batch_size
		self.start_training = args.buffer_start

	def save(self, episodes=None, path=None): # alternatively returns state dict of net
		checkpoint = {
			'alg': 'DDPG',
			'state_dict': self.model.state_dict(),
			'actor_optim': self.actor_optimizer.state_dict(), 
			'critic_optim': self.critic_optimizer.state_dict(),
		}
		if path is None:
			return checkpoint
		assert episodes is not None
		torch.save(checkpoint, path)

	def load(self, checkpoint=None, path=None):
		if checkpoint is None:
			checkpoint = torch.load(path)
		assert checkpoint['alg'] == 'DDPG', 'not the correct agent - this is a {} agent'.format(checkpoint['alg'])
		self.model.load_state_dict(checkpoint['state_dict'])
		self.target = copy.deepcopy(self.net)
		self.actor_optimizer.load_state_dict(checkpoint['actor_optim'])
		self.critic_optimizer.load_state_dict(checkpoint['critic_optim'])

	def get_value(self, state, action=None):
		if len(state.shape) == 1:
			state = state.reshape(1,-1)
		state = Variable(torch.from_numpy(state).type(self.def_type))
		if action is not None:
			if len(action.shape) == 1:
				action = action.reshape(1,-1)
			action = Variable(torch.from_numpy(action).type(self.def_type))
		value = self.model.get_value(state, action).data.numpy()
		value = value.reshape(-1)
		return value.numpy()

	def get_action(self, state, force_noisy=None):
		if len(state.shape) == 1:
			state = state.reshape(1,-1)
		state = torch.from_numpy(state).type(self.def_type)
		#print(state.size())
		action = self.model.get_action(Variable(state)).data.cpu().numpy()
		if force_noisy or (self.mode == 'train' and force_noisy is None):
			action += np.vstack([self.noise.noise() for _ in action])
		action = action.reshape(-1)
		return action

	def learn(self, state, action, reward, next_state, done=False):

		reward = np.array(reward).reshape(1,1)
		state = state.reshape(1,-1)
		action = action.reshape(1,-1)
		next_state = next_state.reshape(1,-1)
		#done = np.array([1 if done else 0]).reshape(1,1)

		self.memory.add(*([torch.from_numpy(element).type(self.def_type)
						  for element in [state, action, reward, next_state]]
						 + [torch.FloatTensor([not done]).type(self.def_type).unsqueeze(0)]))

		loss = None

		if len(self.memory) > self.start_training: # train one batch
			minibatch = self.memory.get_batch(self.batch_size)
			state_batch = torch.cat([data[0] for data in minibatch], dim=0)
			action_batch = torch.cat([data[1] for data in minibatch], dim=0)
			reward_batch = torch.cat([data[2] for data in minibatch], dim=0)
			next_state_batch = torch.cat([data[3] for data in minibatch], dim=0)
			done_batch = torch.cat([data[4] for data in minibatch], dim=0)

			#print(state_batch.size(), action_batch.size(), reward_batch.size(), next_state_batch.size(), done_batch.size())

			# calculate y_batch from targets
			value_batch = self.target.get_value(Variable(next_state_batch)).data
			y_batch = reward_batch + self.discount * value_batch * done_batch

			# optimize critic net 1 step
			critic_loss = self.criterion(self.model.get_value(Variable(state_batch), Variable(action_batch)), Variable(y_batch))
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			actor_loss = - self.model.get_value(Variable(state_batch)).mean()
			# critic_optimizer.zero_grad() # critic network is used to evaluate value
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.model.critic.parameters(), self.target.critic.parameters()):
				target_param.data.mul_(1 - self.tau)
				target_param.data.add_(self.tau, param.data)

			for param, target_param in zip(self.model.actor.parameters(), self.target.actor.parameters()):
				target_param.data.mul_(1 - self.tau)
				target_param.data.add_(self.tau, param.data)

			loss = critic_loss.data[0]

		if done:
			self.noise.reset()

		return loss

	

