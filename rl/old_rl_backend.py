
class Transition_Buffer(object):
	def __init__(self, state_dim, action_dim, device='cpu', max_size=1000):
		self._max_size = max_size
		self._choices = np.arange(self._max_size)
		self._state_dim = state_dim
		self._action_dim = action_dim
		self._device = device
		self.reset()

	def reset(self):
		self._idx = 0
		self._buffer = torch.empty(self._max_size, self._state_dim*2 + self._action_dim + 2).to(self._device)
		self._N = 0

	def to(self, device):
		self._device = device
		self._buffer = self._buffer.to(self._device)

	def add(self, *transition): # state, action, reward, done, next_state
		self._buffer[self._idx] = torch.cat(transition, -1)
		self._idx += 1
		self._idx %= len(self._buffer)
		self._N = min(self._N + 1, len(self._buffer))

	def __len__(self):
		return self._N

	def sample(self, N=1, with_replacement=False):
		assert len(self) and (with_replacement or N <= len(self)), 'not enough samples in replay buffer'
		batch = self._buffer[np.random.choice(self._choices[:len(self)], N, replace=with_replacement)]
		return batch.narrow(-1, 0, self._state_dim), batch.narrow(-1, self._state_dim, self._action_dim), \
		       batch.narrow(-1, self._state_dim+self._action_dim, 1), batch.narrow(-1, self._state_dim+self._action_dim+1, 1), \
		       batch.narrow(-1, self._state_dim + self._action_dim + 2, self._state_dim)

class DDPG(nn.Module):
	def __init__(self, state_dim, action_dim, discount=0.97, actor_steps=1,
	             max_buffer_size=10000, min_buffer_size=None, batch_size=None,
	             tau=0.001, lr=1e-3, weight_decay=1e-4):
		super(DDPG, self).__init__()

		self.model = ActorCritic(state_dim, action_dim)

		self.target_model = copy.deepcopy(self.model)
		self.soft_update = EMA_Update(self.model.parameters(), self.target_model.parameters(), tau=tau)

		self.actor_steps = actor_steps
		self.actor_optim = optim.SGD(self.model.actor.parameters(), lr=lr, weight_decay=weight_decay)
		self.critic_optim = optim.SGD(self.model.critic.parameters(), lr=lr, weight_decay=weight_decay)
		self.criterion = nn.MSELoss()

		self.buffer = Replay_Buffer(state_dim, 1, max_size=max_buffer_size)
		self.discount = discount
		self.min_buffer_size = self.buffer._max_size // 10 if min_buffer_size is None else min_buffer_size
		self.batch_size = self.min_buffer_size // 10 if batch_size is None else batch_size
		assert self.min_buffer_size >= self.batch_size, 'Batch size is too big for this replay buffer'

	def to(self, device):
		self.buffer.to(device)
		super(DDPG, self).to(device)

	def forward(self, state):
		return self.model.get_action(state)

	def learn(self, *transition):

		self.buffer.add(*transition)

		if len(self.buffer) >= self.min_buffer_size:

			state, action, reward, done, next_state = self.buffer.sample(self.batch_size)

			y = reward + self.discount * done * self.target_model(next_state).detach()

			self.critic_optim.zero_grad()
			critic_loss = self.criterion(self.model(state, action), y)
			critic_loss.backward()
			self.critic_optim.step()

			for _ in range(self.actor_steps):
				self.actor_optim.zero_grad()
				value = self.model(state).mean()
				value.mul(-1).backward()
				self.actor_optim.step()

			self.soft_update.step()

			return critic_loss.item(), value.item()

		return None

class Generator(object):
	
	def __init__(self, env_name, policy, default_N=10, T=None, num_workers=0,
				 discount=None, include_last_state=False):
		
		self.pool = None if num_workers == 0 else mp.Pool(num_workers)
		self.default_N = default_N
		self.created = 0
		
		self.env_name = env_name
		self.policy = policy
		
		self.T = T
		self.discount = discount
		self.include_last_state = include_last_state
	
	def __len__(self):
		return self.created
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self.generate()
	
	def generate(self, N=None):
		
		N = self.default_N if N is None else N
		self.created += N
		
		if self.pool is not None:
			return self.pool.starmap(self.rollout, [self.policy] * N)
		
		return map(self.rollout, [self.policy] * N)
	
	def rollout(self, policy):
		
		env = Pytorch_Gym_Env(self.env_name)
		
		T = env.spec.timestep_limit if self.T is None else self.T
		
		states = []
		actions = []
		rewards = []
		
		states.append(env.reset())
		
		for _ in range(T):
			
			actions.append(policy(states[-1]))
			
			state, reward, done, _ = env.step(actions[-1])
			
			states.append(state)
			rewards.append(reward)
			
			if done:
				break
		
		if not self.include_last_state:
			states.pop()
		
		states = torch.stack(states)
		actions = torch.stack(actions)
		rewards = torch.cat(rewards)
		
		if self.discount is not None:
			
			returns = rewards.clone()
			
			for i in range(len(rewards) - 2, -1, -1):
				returns[i] += self.discount * returns[i + 1]
			
			return states, actions, returns
		
		return states, actions, rewards


class Generator(object):
	
	def __init__(self, env_name, policy, T=None,
				 discount=None, include_last_state=False):
		
		self.created = 0
		
		self.env_name = env_name
		self.policy = policy
		
		self.T = T
		self.discount = discount
		self.include_last_state = include_last_state
	
	def __len__(self):
		return self.created
	
	def __call__(self, _):
		
		env = Pytorch_Gym_Env(self.env_name)
		
		T = env.spec.timestep_limit if self.T is None else self.T
		
		states = []
		actions = []
		rewards = []
		
		states.append(env.reset())
		
		for _ in range(T):
			
			actions.append(self.policy(states[-1]))
			
			state, reward, done, _ = env.step(actions[-1])
			
			states.append(state)
			rewards.append(reward)
			
			if done:
				break
		
		if not self.include_last_state:
			states.pop()
		
		states = torch.stack(states)
		actions = torch.stack(actions)
		rewards = torch.cat(rewards)
		
		self.created += 1
		
		if self.discount is not None:
			
			returns = rewards.clone()
			
			for i in range(len(rewards) - 2, -1, -1):
				returns[i] += self.discount * returns[i + 1]
			
			return states, actions, returns
		
		return states, actions, rewards


def run_episodes(mode, N, agent, env, horizon=None):
	train = mode == 'train'
	if train:
		agent.train()
	else:
		agent.eval()

	if horizon is None:
		horizon = env.spec.timestep_limit

	avg_reward = 0
	avg_loss = 0

	for episode in range(N):

		state = env.reset()
		total_reward = 0
		total_loss = 0

		done = False
		step = 0
		while not done and step < horizon:

			action = agent(state).detach()
			next_state, reward, done, _ = env.step(action)

			total_reward += reward.item()

			loss = None
			if train:
				loss = agent.learn(state, action.view(-1).float(), reward, done, next_state)

			state = next_state
			step += 1

			if loss is not None:
				total_loss += loss

		#print('Mode {} Episode {}/{}: reward={:.3f}'.format(mode, episode + 1, N, total_reward))
		avg_reward += total_reward
		avg_loss += total_loss/step

	return avg_reward/N, avg_loss/N


def gen_rollout(env_name, policy, discount=None, T=None, include_last_state=False):
	env = Pytorch_Gym_Env(env_name)
	
	T = env.spec.timestep_limit if T is None else T
	
	states = []
	actions = []
	rewards = []
	
	states.append(env.reset())
	
	for _ in range(T):
		
		actions.append(policy(states[-1]))
		
		state, reward, done, _ = env.step(actions[-1])
		
		states.append(state)
		rewards.append(reward)
		
		if done:
			break
	
	if not include_last_state:
		states.pop()
	
	states = torch.stack(states)
	actions = torch.stack(actions)
	rewards = torch.cat(rewards)
	
	if discount is not None:
		
		returns = rewards.clone()
		
		for i in range(len(rewards) - 2, -1, -1):
			returns[i] += discount * returns[i + 1]
		
		return states, actions, returns
	
	return states, actions, rewards


import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import torch.multiprocessing as mp

import foundation as fd
import foundation.util as util

import gym


def solve(A, b, out=None, bias=True):
	# A: M x N
	# b: N x K
	# x: M x K
	
	if bias:
		A = torch.cat([A, torch.ones(*(A.size()[:-1] + (1,))).type_as(A)], -1)
	
	x, _ = torch.gels(b, A)
	
	if out is None:
		out = nn.Linear(A.size(-1) - 1, b.size(-1), bias=bias).to(A.device)
	
	out.weight.data.copy_(x[:A.size(-1) - 1].t())
	
	if bias:
		out.bias.data.copy_(x[A.size(-1) - 1:A.size(-1), 0])
	
	return out


def run_iteration(mode, N, agent, gen, horizon=None, render=False):
	train = mode == 'train'
	if train:
		agent.train()
	else:
		agent.eval()
	
	states, actions, rewards = zip(*[gen(horizon=horizon, render=render) for _ in range(N)])
	
	loss = None
	if train:
		loss = agent.learn(states, actions, rewards)
	
	reward = sum([r.sum() for r in rewards]) / N
	
	return reward, loss


def MLE(distrib):
	if isinstance(distrib, Normal):
		return distrib.loc
	elif isinstance(distrib, Categorical):
		return distrib.probs.max(-1)[1]
	raise Exception('Distribution {} not recognized (did you forget to add it to MLE function?)'.format(type(distrib)))


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()
		self.net = fd.nets.make_MLP(input_dim=state_dim,
		                            output_dim=action_dim, output_nonlin='softmax',
		                            hidden_dims=[12, 6], nonlin='elu')
	
	def forward(self, state):
		return self.net(state)
	
	def get_policy(self, state):
		return Categorical(self(state))
	
	def get_action(self, state, greedy=None):
		if greedy is None:
			greedy = not self.training
		
		policy = self.get_policy(state)
		return MLE(policy) if greedy else policy.sample()


class QFunction(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(QFunction, self).__init__()
		self.action_dim = action_dim
		self.net = fd.nets.make_MLP(input_dim=state_dim + action_dim,
		                            output_dim=1,
		                            hidden_dims=[12, 6], nonlin='elu')
	
	def forward(self, state, action):
		if action.size(-1) != self.action_dim:
			action = to_one_hot(action, self.action_dim)
		return self.net(torch.cat([state, action], -1))


class ValueFunction(nn.Module):
	def __init__(self, state_dim):
		super(ValueFunction, self).__init__()
		self.net = fd.nets.make_MLP(input_dim=state_dim,
		                            output_dim=1,
		                            hidden_dims=[12, 6], nonlin='elu')
	
	def forward(self, state):
		return self.net(state)


class ActorCritic(nn.Module):  # DDPG
	
	def __init__(self, state_dim, action_dim):
		super(ActorCritic, self).__init__()
		self.actor = Actor(state_dim, action_dim)
		self.critic = QFunction(state_dim, action_dim)
	
	def forward(self, state, action=None):
		if action is None:
			action = self.actor(state)
		return self.critic(state, action)
	
	def get_action(self, state, greedy=None):
		return self.actor.get_action(state, greedy=greedy)


class QNet(nn.Module):  # Action out
	def __init__(self, state_dim, action_dim, epsilon=0.01):
		super(QNet, self).__init__()
		self.epsilon = epsilon  # for epsilon-greedy exploration
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.net = fd.nets.make_MLP(input_dim=state_dim,
		                            output_dim=action_dim,
		                            hidden_dims=[8], nonlin='lrelu')
	
	def forward(self, state, action=None):
		values = self.net(state)
		if action is None:
			return values.max(-1, keepdim=True)  # returns max and argmax
		# print(values.size(), action.size())
		return values.gather(-1, action.long())
	
	def get_action(self, state, greedy=None):
		state = state.view(-1, self.state_dim)
		greedy = not self.training if greedy is None else greedy
		if greedy or np.random.rand() > self.epsilon:
			return self(state)[1]
		return torch.randint(self.action_dim, size=(state.size(0), 1)).to(state.device).long()


class DQN(nn.Module):
	def __init__(self, state_dim, action_dim, discount=0.99, epsilon=0.01,
	             max_buffer_size=2000, min_buffer_size=None, batch_size=None,
	             tau=0.001, use_replica=True, lr=1e-3, weight_decay=1e-4):
		super(DQN, self).__init__()
		
		self.model = QNet(state_dim, action_dim, epsilon)
		
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
		self.buffer.to(device)
		super(DQN, self).to(device)
	
	def forward(self, state):
		return self.model.get_action(state)
	
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


def to_one_hot(idx, max_idx=None):
	if max_idx is None:
		max_idx = idx.max()
	dims = (max_idx,)
	if idx.ndimension() >= 1:
		if idx.size(-1) != 1:
			idx = idx.unsqueeze(-1)
		dims = idx.size()[:-1] + dims
	return torch.zeros(*dims).to(idx.device).scatter_(-1, idx.long(), 1)


class Replay_Buffer(object):
	def __init__(self, max_size=100, device='cpu'):
		self._max_size = max_size
		self._choices = np.arange(self._max_size)
		self._device = device
		self.reset()
	
	def reset(self):
		self._idx = 0
		self._buffer = [None] * self._max_size
		self._lens = [0] * self._max_size
		self._N = 0
	
	def to(self, device):
		self._device = device
		self._buffer = self._buffer.to(self._device)
	
	def add(self, *transition):  # state, action, reward, done, next_state
		self._buffer[self._idx] = torch.cat(transition, -1)
		self._idx += 1
		self._idx %= len(self._buffer)
		self._N = min(self._N + 1, len(self._buffer))
	
	def __len__(self):
		return self._N
	
	def sample(self, N=1, with_replacement=False):
		assert len(self) and (with_replacement or N <= len(self)), 'not enough samples in replay buffer'
		batch = self._buffer[np.random.choice(self._choices[:len(self)], N, replace=with_replacement)]
		return batch.narrow(-1, 0, self._state_dim), batch.narrow(-1, self._state_dim, self._action_dim), \
		       batch.narrow(-1, self._state_dim + self._action_dim, 1), batch.narrow(-1,
		                                                                             self._state_dim + self._action_dim + 1,
		                                                                             1), \
		       batch.narrow(-1, self._state_dim + self._action_dim + 2, self._state_dim)


class EMA_Update(torch.optim.Optimizer):
	
	def __init__(self, source_params, target_params, tau=0.001):
		super(EMA_Update, self).__init__([{'params': source_params},
		                                  {'params': target_params}], defaults={'tau': tau})
	
	def step(self):
		source_group, target_group = self.param_groups
		tau = source_group['tau']
		for source_param, target_param in zip(source_group['params'], target_group['params']):
			target_param.data.mul_(1. - tau)
			target_param.data.add_(tau, source_param.data)


class Pytorch_Gym_Env(object):
	
	def __init__(self, env_name, device='cpu'):
		self._env = gym.make(env_name)
		self.spec = self._env.spec
		self.action_space = self._env.action_space
		self.observation_space = self._env.observation_space
		self._device = device
	
	def reset(self):
		return torch.from_numpy(self._env.reset()).float().to(self._device).view(-1)
	
	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs)
	
	def to(self, device):
		self._device = device
	
	def step(self, action):
		action = action.squeeze().detach().cpu().numpy()
		if action.ndim == 0:
			action = action[()]
		obs, reward, done, info = self._env.step(action)
		obs = torch.from_numpy(obs).float().to(self._device).view(-1)
		reward = torch.tensor(reward).float().to(self._device).view(1)
		done = torch.tensor(done).float().to(self._device).view(1)
		return obs, reward, done, info


class Generator(object):
	
	def __init__(self, env, policy, horizon=None, drop_last_state=True):
		
		self.created = 0
		
		self.policy = policy
		self.env = env
		
		self.drop_last_state = drop_last_state
		
		self.horizon = self.env.spec.timestep_limit if horizon is None else horizon
	
	def __len__(self):
		return self.created
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self()
	
	def __call__(self, horizon=None, render=False):
		
		states = []
		actions = []
		rewards = []
		
		states.append(self.env.reset())
		horizon = self.horizon if horizon is None else horizon
		for _ in range(horizon):
			
			if render:
				self.env.render()
			
			actions.append(self.policy(states[-1]))
			
			state, reward, done, _ = self.env.step(actions[-1])
			
			states.append(state)
			rewards.append(reward)
			
			if done:
				break
		
		if self.drop_last_state:
			states.pop()
		
		states = torch.stack(states)
		actions = torch.stack(actions)
		rewards = torch.cat(rewards)
		
		self.created += 1
		
		return states, actions, rewards

