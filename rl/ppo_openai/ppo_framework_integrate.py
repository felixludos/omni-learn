import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import gym
import numpy as np
import time
import subprocess
import os
import zmq
import sys
import foundation as fd
from foundation import models
from foundation import util
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.nn import functional as F

stack_collate = util.make_collate(stack=True)
cat_collate = util.make_collate(stack=False)

class RunningMeanStd(nn.Module):
	def __init__(self, dim, cmin=-5, cmax=5):
		super().__init__()
		self.dim = dim
		self.n = 0
		self.cmin, self.cmax = cmin, cmax

		self.register_buffer('sum_sq', torch.zeros(dim))
		self.register_buffer('sum', torch.zeros(dim))
		self.register_buffer('mu', torch.zeros(dim))
		self.register_buffer('sigma', torch.ones(dim))

	def update(self, xs):
		xs = xs.view(-1, self.dim)
		self.n += xs.shape[0]
		self.sum += xs.sum(0)
		self.mu = self.sum / self.n

		self.sum_sq += xs.pow(2).sum(0)
		self.mean_sum_sq = self.sum_sq / self.n

		if self.n > 1:
			self.sigma = (self.mean_sum_sq - self.mu**2).sqrt()

	def forward(self, x):
		if self.training:
			self.update(x)
		return ((x - self.mu) / self.sigma).clamp(self.cmin, self.cmax)


class Pytorch_Gym_Env(object):
	'''
	Wrapper for OpenAI-Gym environments so they can easily be used with pytorch
	'''

	def __init__(self, env_name, device='cpu'):
		self._env = gym.make(env_name)
		self._discrete = hasattr(self._env.action_space, 'n')
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
		# action = action
		if not self._discrete:
			action = action.reshape(-1)
		elif action.ndim == 0:
			action = action[()]

		obs, reward, done, info = self._env.step(action)

		obs = torch.from_numpy(obs).float().to(self._device).view(-1)
		reward = torch.tensor(reward).float().to(self._device).view(1)
		done = torch.tensor(done).float().to(self._device).view(1)
		info = {k:torch.tensor(v) for k,v in info.items()}
		return obs, reward, done, info

class _Generator_Iterator(object):
	def __init__(self, gen, step_threshold,
	             step_cutoff, episode_cutoff,
	             step_limit, episode_limit, ):
		self.gen = gen
		self.total_steps = 0
		self.total_episodes = 0

		self.step_threshold = step_threshold
		self.step_cutoff = step_cutoff
		self.episode_cutoff = episode_cutoff

		self.step_limit = step_limit
		self.episode_limit = episode_limit
		self.steps_left = step_limit
		self.episodes_left = episode_limit
		#assert not (self.step_limit is None and self.episode_limit is None), 'must specify limit for iterator'

	def steps_generated(self):
		return self.total_steps
	def episodes_generated(self):
		return self.total_episodes

	def __len__(self):
		l = self.episode_limit if self.episode_limit is not None else self.step_limit
		return 0 if l is None else l # can be infinite

	def __iter__(self):
		return self

	def __next__(self):
		if (self.episodes_left is not None and self.episodes_left <= 0) \
				or (self.steps_left is not None and self.steps_left <= 0):
			raise StopIteration

		step_cutoff = self.steps_left if self.step_cutoff is None else self.step_cutoff
		if self.step_cutoff is not None and self.steps_left is not None:
			step_cutoff = min(step_cutoff, self.step_cutoff)

		episode_cutoff = self.episodes_left if self.episode_cutoff is None else self.episode_cutoff
		if self.episode_cutoff is not None and self.episodes_left is not None:
			episode_cutoff = min(episode_cutoff, self.episode_cutoff)

		E, S, rollouts = self.gen._rollout(step_threshold=self.step_threshold,
		                                   episode_cutoff=episode_cutoff,
		                                   step_cutoff=step_cutoff)
		self.total_episodes += E
		self.total_steps += S
		if self.steps_left is not None:
			self.steps_left -= S
		if self.episodes_left is not None:
			self.episodes_left -= E

		return rollouts

class Generator(object):
	def __init__(self, env, agent, drop_last_state=False, max_episode_length=None,
	             episode_cutoff=None, step_cutoff=None, # hard cutoff per iteration
	             step_threshold=None, # soft cutoff per iteration
	             step_limit=None, episode_limit=None): # overall budget (optional)
		super().__init__()

		self.env = env
		self.agent = agent

		self.drop_last = drop_last_state
		self.step_threshold = step_threshold
		self.episode_cutoff = episode_cutoff
		self.step_cutoff = step_cutoff
		self.max_episode_length = max_episode_length

		self.total_steps = 0
		self.total_episodes = 0

		self.step_limit = step_limit
		self.episode_limit = episode_limit

		assert not (self.step_threshold is None
		            and self.step_cutoff is None
		            and self.episode_cutoff is None), 'must specify iteration size in some way'

	def steps_generated(self):
		return self.total_steps
	def episodes_generated(self):
		return self.total_episodes

	def _rollout(self, step_threshold=None, episode_cutoff=None, step_cutoff=None):
		if step_threshold is None:
			step_threshold = self.step_threshold
		if episode_cutoff is None:
			episode_cutoff = self.episode_cutoff
		if step_cutoff is None:
			step_cutoff = self.step_cutoff

		states = []
		actions = []
		rewards = []
		info = []

		steps_generated = 0
		episodes_generated = 0

		with torch.no_grad():

			while (episode_cutoff is None or episodes_generated < episode_cutoff) \
					and (step_cutoff is None or steps_generated < step_cutoff) \
					and (step_threshold is None or steps_generated < step_threshold):

				S = []
				A = []
				AI = []
				EI = []
				R = []

				S.append(env.reset().view(1, -1))
				episodes_generated += 1

				done = False
				steps = 0
				while not done and (self.max_episode_length is None or steps < self.max_episode_length)\
						and (step_cutoff is None or steps_generated < step_cutoff):
					a, ai = agent.gen_action(S[-1])
					A.append(a)
					AI.append(ai)

					s, r, done, ei = env.step(a)
					steps += 1
					steps_generated += 1

					s = s.view(1, -1)
					EI.append(ei)
					S.append(s)
					R.append(r)

				if self.drop_last:
					S = S[:-1]

				S = torch.cat(S)
				A = torch.cat(A)
				R = torch.stack(R)
				AI = cat_collate(AI)
				EI = cat_collate(EI)

				states.append(S)
				actions.append(A)
				info.append(AI)
				info[-1].update(EI)  # combine all info
				rewards.append(R)

			info = stack_collate(info)
			rollout = {
				'states': stack_collate(states),
				'actions': stack_collate(actions),
				'rewards': stack_collate(rewards),
			}
			rollout.update(info)

		self.total_episodes += episodes_generated
		self.total_steps += steps_generated

		return episodes_generated, steps_generated, rollout

	def __iter__(self):
		return _Generator_Iterator(self, step_threshold=self.step_threshold,
	             step_cutoff=self.step_cutoff, episode_cutoff=self.episode_cutoff,
	             step_limit=self.step_limit, episode_limit=self.episode_limit, )

	def __call__(self, *args, **kwargs):
		return self._rollout(*args, **kwargs)[-1]

def compute_returns(rewards, discount=1.):

	if isinstance(rewards, torch.Tensor):
		rewards = rewards.permute(1,0,2) # iterate over timesteps

		returns = rewards.clone()
		for i in range(returns.size(0)-2, -1, -1):
			returns[i] += discount * returns[i + 1]

		return returns.permute(1,0,2) # switch back to episodes first

	returns = []
	for R in rewards:
		G = R.clone()
		for i in range(G.size(0)-2, -1, -1):
			G[i] += discount * G[i+1]
		returns.append(G)

	return returns

class Model(nn.Module):
	def __init__(self, din, dout):
		super().__init__()
		self.din = din
		self.dout = dout

class Policy(nn.Module):
	def __init__(self, model):
		super().__init__()

		self.model = model

		self.state_dim = self.model.din
		self.action_dim = self.model.dout//2

	def get_action(self, obs):
		return self.get_pi(obs).sample()

	def get_pi(self, x):
		mu, log_sigma = self(x)
		return Normal(mu, log_sigma.exp())

	def forward(self, x):
		y = self.model(x.view(-1,self.model.din))
		dim = y.size(-1)//2
		return y.narrow(-1, 0, dim), y.narrow(-1, dim, dim)

class Baseline(Model):
	def __init__(self, model, optim=None, scheduler=None,
	             optim_type='adam', lr=1e-3, scheduler_lin=None,
	             batch_size=64, epochs_per_step=10, ):
		super().__init__(model.din, model.dout)

		self.batch_size = batch_size
		self.epochs = epochs_per_step

		self.model = model

		self.optim = optim
		if self.optim is None:
			self.optim = util.get_optimizer(optim_type, self.model.parameters(), lr=lr)

		self.scheduler = scheduler
		if self.scheduler is None and scheduler_lin is not None:
			self.scheduler = torch.optim.lr_scheduler.LambdaLR(
				self.optim, lambda x: (scheduler_lin - x) / scheduler_lin,-1)

	def train_step(self, states, returns):

		value_net_losses = []

		self.scheduler.step()

		dataloader = DataLoader(TensorDataset(states, returns),
		                        batch_size=self.batch_size, shuffle=True, num_workers=0)

		for ppo_iter in range(self.epochs):
			for idx, (x, y) in enumerate(dataloader):
				self.optim.zero_grad()
				values = self(x)
				loss = F.mse_loss(values, y)
				loss.backward()

				self.optim.step()

				value_net_losses.append(loss.item())

		return value_net_losses

	def forward(self, x):
		return self.model(x)

class PPO(nn.Module): # PPO using Clipping
	def __init__(self, policy, baseline, discount=1., clip=0.3,
	             optim=None, scheduler=None,
	             optim_type='adam', lr=1e-3, scheduler_lin=None,
	             batch_size=64, epochs_per_step=10, ):
		super().__init__()

		self.policy = policy
		self.baseline = baseline
		self.state_dim, self.action_dim = self.policy.state_dim, self.policy.action_dim
		assert self.baseline.din == self.policy.state_dim, 'policy and baseline dont have the same state space: {} vs {}'.format(self.baseline.din, self.policy.state_dim)

		self.clip = clip
		self.discount = discount
		self.batch_size = batch_size
		self.epochs = epochs_per_step

		self.optim = optim
		if self.optim is None:
			self.optim = util.get_optimizer(optim_type, self.policy.parameters(), lr=lr)

		self.scheduler = scheduler
		if self.scheduler is None and scheduler_lin is not None:
			self.scheduler = torch.optim.lr_scheduler.LambdaLR(
				self.optim, lambda x: (scheduler_lin - x) / scheduler_lin, -1)

	def forward(self, x):
		return self.policy.get_action(x)

	def gen_action(self, x):
		pi = self.policy.get_pi(x)
		action = pi.sample()
		log_prob = pi.log_prob(action)

		return action, {'log_probs': log_prob}

	def train_step(self, states, actions, rewards, log_probs=None, **info):
		returns = compute_returns(rewards, discount=self.discount)

		if not isinstance(states, torch.Tensor):
			states = torch.cat(states)#.detach().view(-1, OBSERVATION_DIM)
		if not isinstance(actions, torch.Tensor):
			actions = torch.cat(actions)#.detach().view(-1, ACTION_DIM)
		if not isinstance(returns, torch.Tensor):
			returns = torch.cat(returns)#.detach().view(-1, 1)

		states = states.view(-1, self.state_dim)
		actions = actions.view(-1, self.action_dim)
		returns = returns.contiguous().view(-1, 1)

		if log_probs is None:
			with torch.no_grad():
				pi = self.policy.get_pi(states)
				log_probs = pi.log_prob(actions)
		else:
			if not isinstance(log_probs, torch.Tensor):
				log_probs = torch.cat(log_probs)#.detach().view(-1, ACTION_DIM)
			log_probs = log_probs.view(-1, self.action_dim)

		self.scheduler.step()
		policy_net_losses = []

		advantages = returns - self.baseline(states).detach()
		advantages = (advantages - advantages.mean()) / advantages.std()

		# print(states.shape, actions.shape, advantages.shape, log_probs.shape)
		# quit()

		dataloader = DataLoader(TensorDataset(states, actions, advantages, log_probs),
		                        batch_size=self.batch_size, shuffle=True, num_workers=0)

		for ppo_iter in range(self.epochs):

			iter_time = time.time()

			for idx, (sample_obs, sample_actions, sample_adv, sample_action_logprobs) in enumerate(
					dataloader):
				self.optim.zero_grad()

				pi = self.policy.get_pi(sample_obs)

				action_logprobs = pi.log_prob(sample_actions)

				r = torch.exp(action_logprobs - sample_action_logprobs)

				surrogate = torch.min(r * sample_adv, r.clamp(1.0 - self.clip, 1.0 + self.clip) * sample_adv)

				policy_net_loss = -surrogate
				policy_net_loss = torch.mean(policy_net_loss)
				policy_net_loss.backward()

				self.optim.step()

				policy_net_losses.append(policy_net_loss.cpu().item())

		baseline_stats = self.baseline.train_step(states, returns)

		return policy_net_losses, baseline_stats

def train(gen, agent, log_interval=None, history_len=None):

	plt.ion()
	fig, ax = plt.subplots()

	last_log_time = time.time()

	gen = iter(gen)

	from collections import deque
	history_buffer = deque([], maxlen=history_len)
	history = []

	def compute_history_stats(itr):
		l = list(history_buffer)
		mean_score = np.mean(l)
		print('%d last %d trials max %.1f min %.1f avg %.1f' % (itr, len(l), max(l), min(l), mean_score))
		history.append((gen.total_steps, mean_score))
		xs, ys = zip(*history)
		ax.clear()
		ax.plot(xs, ys, marker='o')
		plt.pause(0.001)

	agent.to(device=ROLLOUT_DEVICE).eval()

	for train_iter, rollouts in enumerate(gen):

		history_buffer.extend(r.sum().item() for r in rollouts['rewards'])

		start_time = time.time()

		agent.to(device=TRAIN_DEVICE).train()

		policy_net_losses = []
		value_net_losses = []

		stats = agent.train_step(**rollouts)

		if log_interval is not None and train_iter % log_interval == 0:
			policy_net_losses, value_net_losses = stats
			print('%.2f sec/batch   policy %8.2f value %8.2f lr %8.6f' % (
				(time.time() - last_log_time) / log_interval,
				np.mean(policy_net_losses),
				np.mean(value_net_losses),
				agent.scheduler.get_lr()[0]))
				# policy_net_scheduler.get_lr()[0]))
			last_log_time = time.time()

		#total_steps += N

		compute_history_stats(train_iter)

		agent.to(device=ROLLOUT_DEVICE).eval()

	fig.savefig('%s' % FIGURE_NAME)


class NormalizedMLP(Model):
	def __init__(self, input_dim, output_dim, **args):
		super().__init__(input_dim, output_dim)
		self.norm = RunningMeanStd(input_dim)
		self.net = models.make_MLP(input_dim, output_dim, **args)

	def forward(self, x):
		return self.net(self.norm(x))

if __name__ == '__main__':
	ENV_NAME = sys.argv[1]
	FIGURE_NAME = sys.argv[2]

	TRAIN_DEVICE = torch.device('cpu')
	ROLLOUT_DEVICE = torch.device('cpu')

	PPO_CLIP = 0.2
	PPO_EPOCH = 10
	PPO_BATCH = 64

	NORMALIZE_STATE = True
	NORMALIZE_ADV = True

	GAE_LAMBDA = 0.95
	GAMMA = 0.99
	OPTIM_TYPE = 'adam'
	LEARNING_RATE = 3e-4
	N_ENVS = 1
	N_STEPS = 2048
	MAX_STEPS = 10 ** 6
	LOG_INTERVAL = 10
	HISTORY_LEN = 100
	n_batch = MAX_STEPS // (N_STEPS * N_ENVS)

	SEED = 10

	torch.manual_seed(SEED)

	env = Pytorch_Gym_Env(ENV_NAME)
	env._env.seed(SEED)

	OBSERVATION_DIM, ACTION_DIM = len(env.observation_space.low), len(env.action_space.low)

	model_fn = models.make_MLP
	if NORMALIZE_STATE:
		model_fn = NormalizedMLP


	baseline_model = model_fn(OBSERVATION_DIM, 1, hidden_dims=[8, 8], nonlin='prelu')
	baseline = Baseline(baseline_model,
	                    optim_type=OPTIM_TYPE, lr=LEARNING_RATE, scheduler_lin=n_batch,
	                    batch_size=PPO_BATCH, epochs_per_step=PPO_EPOCH, )

	policy_model = model_fn(OBSERVATION_DIM, 2 * ACTION_DIM, hidden_dims=[8, 8], nonlin='prelu')
	policy = Policy(policy_model, )

	agent = PPO(policy, baseline, clip=PPO_CLIP,
	            optim_type=OPTIM_TYPE, lr=LEARNING_RATE, scheduler_lin=n_batch,
	            batch_size=PPO_BATCH, epochs_per_step=PPO_EPOCH,
	            )

	gen = Generator(env, agent, step_limit=1000000,
	                step_threshold=N_STEPS, drop_last_state=True)

	train(gen, agent, log_interval=LOG_INTERVAL, history_len=HISTORY_LEN)#, policy_net_opt, value_net_opt)
