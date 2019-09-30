
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from .framework import Agent, ActorCriticAgent
from .utils import MLE, compute_returns, EMA_Update

import foundation as fd
import foundation.util as util
from foundation import nets
import gym

class REINFORCE(Agent):
	
	def __init__(self, policy, discount=0.97, optim_type='adam', lr=1e-3, weight_decay=1e-4):
		super(REINFORCE, self).__init__(policy, discount=discount)
		
		self.baseline = nn.Linear(self.policy.state_dim, 1)
		
		self.optim = nets.get_optimizer(optim_type, self.policy.parameters(), lr=lr, weight_decay=weight_decay)

		self.stats.new('policy-loss', 'bsln-error')
	
	def learn(self, states, actions, rewards):

		returns = [compute_returns(rs, discount=self.discount) for rs in rewards]
		states, actions, returns = torch.cat(states), torch.cat(actions), torch.cat(returns)

		advantages = returns - self.baseline(states).squeeze()
		
		pi = self.policy.get_pi(states)
		
		self.optim.zero_grad()
		perf = -advantages * pi.log_prob(actions)
		perf = perf.mean()
		perf.backward()
		self.optim.step()

		error = F.mse_loss(self.baseline(states).squeeze(), returns).detach()

		util.solve(states, returns.unsqueeze(-1), out=self.baseline)

		self.stats.update('policy-loss', perf.detach())
		self.stats.update('bsln-error', error)
		
		return self.stats # Returns a rough estimate of the error in the baseline (dont worry about this too much)


class PPO(Agent):
	
	def __init__(self, policy, discount=0.97, optim_type='adam', lr=1e-3, weight_decay=1e-4,
	             clipping=None, target_kl=None, kl_weight=1., epochs=1, batch_size=32, ):
		super(PPO, self).__init__(policy, discount=discount)
		
		self.epochs = epochs
		self.batch_size = batch_size
		
		self.clipping = clipping
		self.beta = kl_weight
		self.target_kl = target_kl
		
		self.baseline = nn.Linear(self.policy.state_dim, 1)
		
		self.optim = nets.get_optimizer(optim_type, self.policy.parameters(), lr=lr, weight_decay=weight_decay)
		
		self.stats.new('perf', 'bsln-mse', 'kl-div', 'cpi', 'beta')
	
	def learn(self, states, actions, rewards):
		returns = [compute_returns(rs, discount=self.discount) for rs in rewards]
		states, actions, returns = torch.cat(states), torch.cat(actions), torch.cat(returns)
		
		values = self.baseline(states).detach().squeeze()
		advantages = returns - values
		
		old_policy = copy.deepcopy(self.policy)
		
		batch_size = self.batch_size
		if batch_size is None:
			batch_size = states.size(0)
		
		loader = DataLoader(TensorDataset(states, actions, advantages),
		                    batch_size=batch_size, shuffle=True)
		
		for epoch in range(self.epochs):
			for s, a, v in loader:
				
				pi, pi_old = self.policy.get_pi(s), old_policy.get_pi(s, include_old=True)[1]
				
				cpi = (pi.log_prob(a) - pi_old.log_prob(a)).exp()
				
				vcpi = v*cpi
				if self.clipping is not None:
					vcpi = torch.min(vcpi, v*cpi.clamp(1-self.clipping, 1+self.clipping))
		
				cpi = vcpi.mean()
				kl = distrib.kl_divergence(pi_old, pi).mean()
		
				self.optim.zero_grad()
				perf = cpi - self.beta * kl
				(-perf).backward() # maximize perf
				self.optim.step()
				
				if self.target_kl is not None:
					if kl < self.target_kl/1.5:
						self.beta *= 0.5
					if kl > self.target_kl*1.5:
						self.beta *= 2
						
				self.stats.update('cpi', cpi.detach())
				self.stats.update('kl-div', kl.detach())
				self.stats.update('perf', perf.detach())
				self.stats.update('beta', self.beta)
				
		
		error = F.mse_loss(values, returns).detach()
		util.solve(states, returns.unsqueeze(-1), out=self.baseline)
		
		self.stats.update('bsln-mse', error)
		
		return self.stats

class NPG(Agent):
	
	def __init__(self, policy, discount=0.97, optim_type='adam', lr=1e-3, weight_decay=1e-4):
		super(NPG, self).__init__(policy, discount=discount)
		
		self.baseline = nn.Linear(self.policy.state_dim, 1)
		
		self.optim = nets.get_optimizer(optim_type, self.policy.parameters(), lr=lr, weight_decay=weight_decay)
		
		self.stats.new('policy-loss', 'bsln-error', 'vpg-norm', 'npg-norm')
		
	def learn(self, states, actions, rewards):
		
		returns = [compute_returns(rs, discount=self.discount) for rs in rewards]
		states, actions, returns = torch.cat(states), torch.cat(actions), torch.cat(returns)
		
		advantages = returns - self.baseline(states).squeeze()
		
		pi, pi_old = self.policy.get_pi(states, include_old=False)
		
		self.optim.zero_grad()
		perf = -advantages * pi.log_prob(actions)
		perf = perf.mean()
		perf.backward()
		self.optim.step()
		
		error = F.mse_loss(self.baseline(states).squeeze(), returns).detach()
		
		util.solve(states, returns.unsqueeze(-1), out=self.baseline)
		
		self.stats.update('policy-loss', perf.detach())
		self.stats.update('bsln-error', error)
		
		return self.stats  # Returns a rough estimate of the error in the baseline (dont worry about this too much)


class A3C(ActorCriticAgent):
	
	def __init__(self, actor, critic, discount=0.97, optim_type='adam', lr=1e-3, weight_decay=1e-4):
		super(A3C, self).__init__(actor=actor, critic=critic, discount=discount)
		
		self.actor_optim = nets.get_optimizer(optim_type, self.actor.parameters(), lr=lr, weight_decay=weight_decay)
		self.critic_optim = nets.get_optimizer(optim_type, self.critic.parameters(), lr=lr, weight_decay=weight_decay)

		self.stats.new('actor-loss', 'critic-loss')
	
	def learn(self, states, actions, rewards):

		returns = [compute_returns(rs, discount=self.discount) for rs in rewards]
		states, actions, returns = torch.cat(states), torch.cat(actions), torch.cat(returns)

		advantages = returns - self.critic(states).squeeze()
		
		pi = self.actor.get_pi(states)
		
		self.actor_optim.zero_grad()
		perf = -advantages.detach() * pi.log_prob(actions)
		perf = perf.mean()
		perf.backward()
		self.actor_optim.step()
		
		self.critic_optim.zero_grad()
		loss = advantages.pow(2).mean()
		loss.backward()
		self.critic_optim.step()

		self.stats.update('actor-loss', perf.detach())
		self.stats.update('critic-loss', loss.detach())
		
		return self.stats


class DQN(Agent):
	def __init__(self, policy, discount=0.99, buffer=None, batch_size=None, min_buffer_size=None,
	             tau=0.001, use_replica=True, optim_type='adam', lr=1e-3, weight_decay=1e-4):
		super(DQN, self).__init__(policy, discount=discount)

		self.target = self.policy
		self.soft_update = None
		if use_replica:
			self.target = copy.deepcopy(self.policy)
			self.soft_update = EMA_Update(self.policy.parameters(), self.target.parameters(), tau=tau)

		self.optim = nets.get_optimizer(optim_type, self.policy.parameters(), lr=lr, weight_decay=weight_decay)
		self.criterion = nn.SmoothL1Loss()  # nn.MSELoss()

		self.buffer = buffer
		self.min_buffer_size = min_buffer_size if self.buffer is not None else None
		self.batch_size = batch_size

		self.stats.new('loss')

	def to(self, device):
		if self.buffer is not None:
			self.buffer.to(device)
		return super(DQN, self).to(device)

	def learn(self, states, actions, rewards):

		if self.buffer is not None:
			self.buffer.extend(zip(states, actions, rewards))

		if self.min_buffer_size is None or len(self.buffer) >= self.min_buffer_size:

			loader = DataLoader(self.buffer, shuffle=True, batch_size=self.batch_size)
			loss_stat = util.AverageMeter()

			for state, action, reward, done, next_state in loader:

				y = reward + self.discount * done * self.target(next_state)[0].detach()

				self.optim.zero_grad()
				loss = self.criterion(self.policy(state, action).squeeze(), y)
				loss.backward()
				self.optim.step()

				if self.soft_update is not None:
					self.soft_update.step()

				loss_stat.update(loss.detach())

			self.stats.update('loss', loss_stat.avg)

		return self.stats


class DDQN(Agent): # double DQN
	def __init__(self, policy, discount=0.99, epsilon=0.01, buffer=None, batch_size=None,
	             tau=0.001, use_replica=True, optim_type='adam', lr=1e-3, weight_decay=1e-4):
		raise NotImplementedError
		super(DQN, self).__init__(policy, discount=discount)

		self.target_model = self.model
		self.soft_update = None
		if use_replica:
			self.target_model = copy.deepcopy(self.model)
			self.soft_update = EMA_Update(self.model.parameters(), self.target_model.parameters(), tau=tau)

		self.optim = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
		self.criterion = nn.SmoothL1Loss()  # nn.MSELoss()

		self.buffer = Replay_Buffer(state_dim, 1)
		self.discount = discount
		self.min_buffer_size = self.buffer._max_size // 10 if min_buffer_size is None else min_buffer_size
		self.batch_size = self.min_buffer_size // 10 if batch_size is None else batch_size
		assert self.min_buffer_size >= self.batch_size, 'Batch size is too big for this replay buffer'

	def to(self, device):
		if self.buffer is not None:
			self.buffer.to(device)
		return super(DQN, self).to(device)

	def learn(self, *transition):

		self.buffer.add(*transition)

		if len(self.buffer) >= self.min_buffer_size:

			state, action, reward, done, next_state = self.buffer.sample(self.batch_size)

			y = reward + self.discount * done * self.target_model(next_state)[0].detach()

			# print(reward.size(), done.size(), self.target_model(next_state)[0].detach().size())

			self.optim.zero_grad()
			# print(self.model(state, action).size(), y.size())
			loss = self.criterion(self.model(state, action), y)
			loss.backward()
			self.optim.step()

			if self.soft_update is not None:
				self.soft_update.step()

			return loss.item()

		return None

class DDPG(ActorCriticAgent):
	def __init__(self, policy, discount=0.97, actor_steps=1, buffer=None,
	             min_buffer_size=None, batch_size=None, tau=0.001,
	             optim_type='adam', lr=1e-3, weight_decay=1e-4):
		super(DDPG, self).__init__(policy.actor, policy.critic, discount=discount)

		self.target = copy.deepcopy(policy)
		self.soft_update = EMA_Update(policy.parameters(),
									  self.target.parameters(),
									  tau=tau)

		self.actor_steps = actor_steps
		self.actor_optim = nets.get_optimizer(optim_type, self.actor.parameters(), lr=lr, weight_decay=weight_decay)
		self.critic_optim = nets.get_optimizer(optim_type, self.critic.parameters(), lr=lr, weight_decay=weight_decay)
		self.criterion = nn.MSELoss()
		
		self.stats.new('actor-value', 'critic-loss')
		
		self.buffer = buffer
		self.min_buffer_size = min_buffer_size if self.buffer is not None else None
		self.batch_size = batch_size
		
	def to(self, device):
		self.buffer.to(device)
		super(DDPG, self).to(device)

	def learn(self, states, actions, rewards):

		#print(states[0].size(), actions[0].size(), rewards[0].size())
		
		if self.buffer is not None:
			self.buffer.extend(zip(states, actions, rewards))

		if self.min_buffer_size is None or len(self.buffer) >= self.min_buffer_size:

			loader = DataLoader(self.buffer, shuffle=True, batch_size=self.batch_size)
			critic_loss_stat, actor_loss_stat = util.AverageMeter(), util.AverageMeter()

			for state, action, reward, done, next_state in loader:
				
				y = reward + self.discount * done * self.target(next_state).detach().squeeze()

				self.critic_optim.zero_grad()
				critic_loss = self.criterion(self.critic(state, action).squeeze(), y.squeeze())
				critic_loss.backward()
				self.critic_optim.step()
				
				critic_loss_stat.update(critic_loss.detach())
				
				for _ in range(self.actor_steps):
					self.actor_optim.zero_grad()
					value = self.critic(state, self.actor.get_action(state, greedy=True)).mean()
					value.mul(-1).backward()
					self.actor_optim.step()
					
					actor_loss_stat.update(value.detach())
				
				self.soft_update.step()
				
			self.stats.update('critic-loss', critic_loss_stat.avg)
			self.stats.update('actor-value', actor_loss_stat.avg)

		return self.stats