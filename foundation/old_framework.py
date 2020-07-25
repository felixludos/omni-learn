
import sys, os
import numpy as np
import torch
import copy
import torch.nn as nn
from . import util
import torch.multiprocessing as mp
from torch.utils.data.dataloader import ExceptionWrapper
from itertools import chain
from .models.probs import MLE

class Agent(object):
	def __init__(self, policy, baseline=None, discount=0.99): # contains an optimizer
		self.policy = policy
		self.baseline = baseline
		if discount is not None:
			self.discount = discount
	
	def train_step(self, paths):  # paths is a single NS containing lists of tensors
		
		if 'returns' not in paths:
			paths.returns = self.compute_returns(paths.rewards)
		if 'advantages' not in paths:
			paths.advantages = self.compute_advantages(paths.returns, paths.states)
		
		# train baseline
		baseline_stats = None
		if self.baseline is not None:
			baseline_stats = self.baseline.train_step(paths.states, paths.returns)
		
		# fully collate
		if isinstance(paths.returns, list):
			paths.returns = torch.cat(paths.returns)
		if isinstance(paths.advantages, list):
			paths.advantages = torch.cat(paths.advantages)
		if isinstance(paths.states, list):
			paths.states = torch.cat(paths.states)
		if isinstance(paths.actions, list):
			paths.actions = torch.cat(paths.actions)
		
		update_stats = self._update_policy(paths)
		
		if baseline_stats is not None:
			update_stats.join(baseline_stats, prefix='bsln-')
		
		return update_stats
		
	def _update_policy(self, paths):
		'''
		
		:param paths: NS containing batch of trajectories
		:return:
		'''
		pass
	
	def train(self):
		self.policy.train()
	
	def eval(self):
		self.policy.eval()
		
	def state_dict(self): # returns dict of all info to be saved (including policy, baseline)
		state_dict = {'policy': self.policy.state_dict(),}
		if self.baseline is not None:
			state_dict['baseline'] = self.baseline.state_dict()
		return state_dict
	
	def load_state_dict(self, state_dict):
		self.policy.load_state_dict(state_dict['policy'])
		if self.baseline is not None and 'baseline' in state_dict:
			self.baseline.load_state_dict(state_dict['baseline'])
	
	def compute_advantages(self, returns, obs):
		'''

		:param returns: computed returns ([B] x T x 1)
		:param obs: ([B] x T x O)
		:param baseline: callable(T x O) - returns (T x 1)
		:return:
		'''
		
		if self.baseline is None:
			return returns
		
		return [r - self.baseline(o) for r, o in zip(returns, obs)]
	
	def compute_returns(self, rewards):  # rewards [B] x T x 1
		full = []
		for sample in rewards:
			returns = []
			run_sum = 0
			for i in range(sample.size(0)-1,-1,-1):
				run_sum = sample[i] + self.discount * run_sum
				returns.append(run_sum)
			full.append(torch.stack(returns[::-1]))
			
		return full  # [B] x T x 1

def _agent_training_backend(in_queue, out_queue):
	agent = None
	while True:
		
		paths = in_queue.get()
		if paths is None:
			break
		try:
			if isinstance(paths, Agent):
				agent = paths  # set agent
			else:
				out_queue.put(agent.train_step(paths))
		except Exception:
			out_queue.put(ExceptionWrapper(sys.exc_info()))
class Parallel_Agent(Agent):  # wrapper class for agents, executing train_step in a different process
	def __init__(self, agent, block=True):
		super(Parallel_Agent, self).__init__(agent.policy, agent.baseline, agent.discount)
		self.block = block  # flag to block when training to guarantee training is complete before returning from train_step method
		# remove remote baseline
		agent.baseline = None
		
		self.inq = mp.Queue()
		self.outq = mp.Queue()
		
		self.backend = mp.Process(target=_agent_training_backend, args=(self.inq, self.outq))
		self.backend.daemon = True
		self.backend.start()
		
		self.inq.put(agent)  # send agent to initialize backend
	
	def _dispatch_train_step(self, paths):
		
		if 'returns' not in paths:
			paths.returns = self.compute_returns(paths.rewards)
		if 'advantages' not in paths:
			paths.advantages = self.compute_advantages(paths.returns, paths.states)
		
		collated_paths = util.TreeSpace()
		if isinstance(paths.returns, list):
			collated_paths.returns = torch.cat(paths.returns)
		if isinstance(paths.advantages, list):
			collated_paths.advantages = torch.cat(paths.advantages)
		if isinstance(paths.states, list):
			collated_paths.states = torch.cat(paths.states)
		if isinstance(paths.actions, list):
			collated_paths.actions = torch.cat(paths.actions)
		
		# dispatch policy update
		self.last_stats = None
		self.baseline_stats = None
		self.inq.put(collated_paths)
		
		# train baseline while policy is learning
		if self.baseline is not None:
			self.baseline_stats = self.baseline.train_step(paths)
		
		if self.block:  # wait until policy update is complete
			return self._get_stats()
		
		return self.baseline_stats
	
	def _get_stats(self):
		if self.last_stats is None:
			update_stats = self.outq.get(timeout=10)
			if isinstance(update_stats, ExceptionWrapper):
				raise update_stats.exc_type(update_stats.exc_msg)
			
			if self.baseline_stats is not None:
				update_stats.join(self.baseline_stats, prefix='bsln-')
			self.last_stats = update_stats
		return self.last_stats
	
	def __del__(self):
		self.inq.put(None)  # shutdown backend
	
	def __getattribute__(self, item):
		if item == 'train_step':  # only dispatch
			return self._dispatch_train_step
		return super(Parallel_Agent, self).__getattribute__(item)

class Manager(Agent):
	def __init__(self, agents, parallel, blocking, separate_stats=False):
		super(Manager, self).__init__(self.full_policy, baseline=None, discount=None)
		
		assert len(parallel) == len(blocking) == len(agents), 'must be same length'
		self.agents = [(Parallel_Agent(agent, block=b) if p else agent) for agent, p, b in zip(agents, parallel, blocking)]
		
		self.blocking = [(b if p else None) for p, b in zip(parallel, blocking)]
		
		self.separate_stats = separate_stats

	def full_policy(self, obs): # list of obs
		return [agent.policy(o)[0][0] for agent, o in zip(self.agents, obs)], {}
		
	def train_step(self, paths):
		
		results = [agent.train_step(p) for agent, p in zip(self.agents, paths)] # dispatch/train
		
		return self._collect_stats(results)

	def _collect_stats(self, results=None):
		
		stats = [(res if b != False else agent._get_stats()) for agent, res, b in
		         zip(self.agents, results, self.blocking)]
		
		if self.separate_stats:
			joined_stats = util.StatsMeter()
			for i, stat in enumerate(stats):
				joined_stats.join(stat, prefix='a{}'.format(i))
		else:
			joined_stats = stats[0]
			for s in stats[1:]:
				joined_stats.join(s)
		
		return joined_stats

	def state_dict(self):
		return [agent.state_dict() for agent in self.agents]
	
	def load_state_dict(self, state_dict):
		for agent, state in zip(self.agents, state_dict):
			agent.load_state_dict(state)

	def train(self):
		for agent in self.agents:
			agent.train()
	
	def eval(self):
		for agent in self.agents:
			agent.eval()

class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
	
	def get_pi(self, obs):
		raise Exception('not overridden')

class Baseline(object):
	def __init__(self):
		pass
	
	def train_step(self, paths):
		pass
	
	def state_dict(self):
		pass
	
	def load_state_dict(self, state_dict):
		pass
	
	def __call__(self, obs): # single path T x O
		pass

class Env(object):
	def __init__(self, spec, ID=None):
		self.env_id = ID
		self.spec = spec
		self.seed()
		self._horizon = self.spec.horizon

	@property
	def horizon(self):
		return self._horizon

	def seed(self, seed=None):
		self._seed = np.random.randint(2**32) if seed is None else seed
		self._tgen = torch.manual_seed(self._seed)
		self._npgen = np.random.RandomState(self._seed)
		return self._seed

	def reset(self, init_state=None): # returns: obs
		raise NotImplementedError

	def step(self, action): # returns: next_obs, reward, done, info
		raise NotImplementedError

	def render(self, *args, **kwargs):
		raise NotImplementedError

	def evaluate_policy(self, N, policy, T=None, init_state=None, seed=None, render=False):

		if seed is not None:
			self.seed(seed)
		T = self.horizon if T is None else min(T, self.horizon)

		stats = util.StatsMeter('perf', 'len')

		for ep in range(N):

			total = 0
			# name = 'ep_%i_rewards' % ep
			# stats.new(name)

			if init_state is not None:
				o = self.reset(init_state)
			else:
				o = self.reset()

			for step in range(T):
				if render:
					self.render()
				a = policy(o)[0]
				o, r, done, info = self.step(a)
				for k, v in info.items():
					if k not in stats:
						stats.new(k)
					stats.update(k, v)
				total += r[0] if isinstance(r, list) else r
				if done: break

			# stats.update('perf', stats[name].sum)
			stats.update('perf', total)  # avg reward per step
			stats.update('len', step+1)

		return stats

	def visualize_policy(self, N, policy, T=None):
		return self.evaluate_policy(N, policy, T=T, render=True)

class BatchedEnv(Env):
	def __init__(self, batch_size=1, spec=None, ID=None):
		self.batch_size = batch_size
		super(BatchedEnv, self).__init__(spec, ID)

class Model(nn.Module):  # any vector function (MLP, linear...)
	def __init__(self, in_dim, out_dim):
		super(Model, self).__init__()
		self.din = in_dim
		self.dout = out_dim
		self.device = 'cpu'

	def cuda(self):
		self.device = 'cuda'
		super(Model, self).cuda()

	def cpu(self):
		self.device = 'cpu'
		super(Model, self).cpu()

	def to(self, device):
		self.device = device
		super(Model, self).to(device)

	def forward(self, x):
		return x  # Identity model

	def get_loss(self, *args, stats=None, **kwargs):
		raise NotImplementedError

class Unsupervised_Model(Model):
	def __init__(self, criterion, in_dim, out_dim=1):
		super(Unsupervised_Model, self).__init__(in_dim, out_dim)
		self.criterion = criterion

	def get_loss(self, x, stats=None, **kwargs):
		raise NotImplementedError

class Supervised_Model(Model):
	def __init__(self, criterion, in_dim, out_dim):
		super(Supervised_Model, self).__init__(in_dim, out_dim)
		self.criterion = criterion

	def get_loss(self, x, y, stats=None, **kwargs):
		return self.criterion(self(x), y)

class Generative_Model(Model):

	def generate(self, N=1):
		raise NotImplementedError

class Value_Function(Supervised_Model):
	def __init__(self, criterion, in_dim, out_dim=1):
		super(Value_Function, self).__init__(criterion, in_dim, out_dim)

class EncoderDecoder(Model):

	def encode(self, observed_state):
		raise NotImplementedError

	def decode(self, latent_state):
		raise NotImplementedError

class Transition_Model(Model):
	def __init__(self, state_dim, ctrl_dim):
		super(Transition_Model, self).__init__(state_dim+ctrl_dim, state_dim)

		self.state_dim = state_dim
		self.ctrl_dim = ctrl_dim

		self.info = util.TreeSpace()

	def sequence(self, state0, ctrls, ret_info=False, ret_all=True):

		states = [state0]
		dynamics = util.TreeSpace(**{k:[] for k in self.info})

		for ctrl in ctrls:
			states.append( self( states[-1], ctrl) )

			if ret_info:
				for k in self.info:
					dynamics[k].append(self.info[k])

		if ret_all:
			output = torch.stack(states[1:]) # T x B x S
		else:
			output = states[-1]

		if ret_info:
			return output, dynamics
		return output

	def jacobians(self, x, u): # by finite differencing

		deltas_x = torch.eye(self.state_dim).cuda() * self.eps
		deltas_u = torch.eye(self.ctrl_dim).cuda() * self.eps

		x = x.view(1,self.state_dim)
		u = u.view(1, self.ctrl_dim)

		x_inputs = torch.cat([deltas_x+x] + [x]*(self.ctrl_dim+1))
		u_inputs = torch.cat([u] * (self.state_dim + 1) + [deltas_u+u])

		output = self(x_inputs, u_inputs)

		Jx, n, Ju = output.narrow(0, 0, self.state_dim), \
					output.narrow(0, self.state_dim, 1), \
					output.narrow(0, self.state_dim+1, self.ctrl_dim)

		return (Jx - n) / self.eps, (Ju - n) / self.eps # returns transposed jacobians

	def forward(self, state, ctrl): # identity model
		return state

class Inverse_Model(Model):
	def __init__(self, state_dim, ctrl_dim):
		super(Inverse_Model, self).__init__(state_dim * 2, ctrl_dim)
	
	def forward(self, state, next_state):
		raise NotImplementedError

class Exploration(object):
	def __init__(self, action_dim):
		self.dout = action_dim
		
	def __call__(self, pi): # gets policy distribution
		raise Exception('not overridden') # returns action
	
	def reset(self):
		pass