
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from .utils import MLE, to_one_hot
from .framework import Policy

import foundation
from foundation import nets

class ValueFunction(nn.Module):

	def __init__(self, state_dim, hidden_dims=[], nonlin='elu'):
		super(ValueFunction, self).__init__()
		self.state_dim = state_dim
		self.net = nets.make_MLP(input_dim=state_dim, output_dim=1,
		                         hidden_dims=hidden_dims, nonlin=nonlin)

	def forward(self, state):
		return self.net(state)

class QFunction(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[], nonlin='elu'):
		super(QFunction, self).__init__()
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.net = nets.make_MLP(input_dim=state_dim+action_dim, output_dim=1,
		                         hidden_dims=hidden_dims, nonlin=nonlin)

	def forward(self, state, action):
		#if 'Float' not in action.type():
		if action.size(-1) != self.action_dim:
			action = to_one_hot(action, self.action_dim)
		return self.net(torch.cat([state, action], -1))

class ActionOut_QFunction(Policy):
	def __init__(self, state_dim, action_dim, hidden_dims=[], nonlin='elu', temperature=1, epsilon=0.01):
		super(ActionOut_QFunction, self).__init__(state_dim, action_dim)
		self.net = nets.make_MLP(input_dim=state_dim, output_dim=action_dim,
		                         hidden_dims=hidden_dims, nonlin=nonlin)

		self.temperature = temperature
		self.epsilon = epsilon

	def forward(self, state, action=None):
		values = self.net(state)
		if action is None:
			return values.max(-1)
		return values.gather(-1, action.long())

	def get_action(self, state, greedy=None):
		state = state.view(-1, self.state_dim)
		greedy = not self.training if greedy is None else greedy
		if greedy or np.random.rand() > self.epsilon:
			return self(state)[1]
		return torch.randint(self.action_dim, size=(state.size(0),)).to(state.device).long()

	def get_pi(self, state, include_old=False):
		if False:
			values = self(state)
			values -= values.mean()
			values *= self.temperature

			pi = Categorical(logits=values)
			if include_old:
				pi_old = Categorical(logits=values.detach())
				return pi, pi_old

			return pi
		raise Exception('Deterministic policies have no distribution')

class Discrete_Policy(Policy):
	def __init__(self, state_dim, action_dim, hidden_dims=[], nonlin='elu'):
		super(Discrete_Policy, self).__init__(state_dim, action_dim)

		self.net = nets.make_MLP(input_dim=state_dim, output_dim=action_dim,
		                         hidden_dims=hidden_dims, nonlin=nonlin, output_nonlin='softmax')

	def forward(self, state):
		return self.net(state)

	def get_pi(self, state, include_old=False):

		probs = self(state)

		pi = Categorical(probs)
		if include_old:
			pi_old = Categorical(probs.detach())
			return pi, pi_old

		return pi


class Gaussian_Policy(Policy):
	def __init__(self, state_dim, action_dim, hidden_dims=[], nonlin='elu',
	             init_log_std=0, min_log_std=-3):
		super(Gaussian_Policy, self).__init__(state_dim, action_dim)

		self.net = nets.make_MLP(input_dim=state_dim, output_dim=action_dim,
		                         hidden_dims=hidden_dims, nonlin=nonlin)

		self.log_std = nn.Parameter(torch.ones(action_dim)*init_log_std, requires_grad=True)
		self.min_log_std = min_log_std

	def forward(self, state):
		return self.net(state)

	def get_pi(self, state, include_old=False):
		mu = self(state)
		sigma = self.log_std.clamp(min=self.min_log_std).exp()

		if mu.ndimension() == 2:
			sigma = sigma.unsqueeze(0)

		pi = Normal(loc=mu, scale=sigma)
		if include_old:
			pi_old = Normal(loc=mu.detach(), scale=sigma.detach())
			return pi, pi_old

		return pi


class Full_Gaussian_Policy(Policy):
	def __init__(self, state_dim, action_dim, hidden_dims=[], nonlin='elu', min_log_std=-3):
		super(Full_Gaussian_Policy, self).__init__(state_dim, action_dim)

		self.net = nets.make_MLP(input_dim=self.state_dim, output_dim=self.action_dim*2,
		                         hidden_dims=hidden_dims, nonlin=nonlin)

		self.min_log_std = min_log_std

	def forward(self, state):
		return self.net(state)

	def get_pi(self, state, include_old=False):
		actions = self(state)

		mu, sigma = actions.narrow(-1, 0, self.action_dim), actions.narrow(-1, self.action_dim, self.action_dim).clamp(min=self.min_log_std).exp()

		pi = Normal(loc=mu, scale=sigma)
		if include_old:
			pi_old = Normal(loc=mu.detach(), scale=sigma.detach())
			return pi, pi_old

		return pi


class ActorCritic(Policy):  # DDPG
	
	def __init__(self, actor, critic):
		assert actor.state_dim == critic.state_dim and actor.action_dim == critic.action_dim
		super(ActorCritic, self).__init__(actor.state_dim, actor.action_dim)
		self.actor = actor
		self.critic = critic
	
	def forward(self, state, action=None):
		if action is None:
			action = self.actor.get_action(state, greedy=True)
		return self.critic(state, action)
	
	def get_pi(self, state, include_old=False):
		return self.actor.get_pi(state, include_old=include_old)
	
	def get_action(self, state, greedy=None):
		return self.actor.get_action(state, greedy=greedy)