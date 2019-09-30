

import torch
import torch.nn as nn

from .utils import MLE
import foundation.util as util

class Policy(nn.Module):

	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

	def get_pi(self, state, include_old=False):
		raise NotImplementedError

	def get_action(self, state, greedy=None):
		greedy = not self.training if greedy is None else greedy

		policy = self.get_pi(state)
		return MLE(policy) if greedy else policy.sample()

class Agent(nn.Module):

	def __init__(self, policy, discount=0.99):
		super(Agent, self).__init__()
		self.stats = util.StatsMeter()
		self.policy = policy
		self.discount = discount

	def forward(self, state):
		return self.policy.get_action(state)

	def learn(self, states, actions, rewards):
		raise NotImplementedError

class ActorCriticAgent(Agent):

	def __init__(self, actor, critic, discount=0.99):
		super(ActorCriticAgent, self).__init__(None, discount=discount)
		self.actor = actor
		self.critic = critic

	def forward(self, state):
		return self.actor.get_action(state)

	def learn(self, states, actions, rewards):
		raise NotImplementedError