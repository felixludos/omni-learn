import sys, os
import numpy as np
import torch
import copy
import torch.nn as nn
from . import util
import torch.multiprocessing as mp
from itertools import chain

class Model(nn.Module):  # any vector function
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.din = in_dim
		self.dout = out_dim
		self.device = 'cpu'

	def cuda(self, device=None):
		self.device = 'cuda' if device is None else device
		super(Model, self).cuda(device)

	def cpu(self):
		self.device = 'cpu'
		super(Model, self).cpu()

	def to(self, device):
		self.device = device
		super(Model, self).to(device)



	def pre_epoch(self): # called at the beginning of each epoch
		pass

	def post_epoch(self, stats=None): # called at the end of each epoch
		pass

class CompositeModel(Model):
	def __init__(self, *models):
		super().__init__(models[0].din, models[-1].dout)
		self.models = nn.ModuleList(models)

	def forward(self, x):
		for m in self.models:
			x = m(x)
		return x


class Generative(object):
	def generate(self, N=1):
		raise NotImplementedError

class Encodable(object):
	def encode(self, x):
		raise NotImplementedError # should output q

class Decodable(object):
	def decode(self, q):
		return NotImplementedError # should output x

class Recordable(Model):
	def __init__(self, *args, stats=None, **kwargs):
		super().__init__(*args, **kwargs)

		if stats is None:
			stats = util.StatsMeter()
		self.stats = stats

	def reset_stats(self):
		self.stats.reset()

	def pre_epoch(self):
		super().pre_epoch()
		self.reset_stats()

class Visualizable(Recordable):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._viz_counter = 0
		self.reset_viz_counter()
	def reset_viz_counter(self):
		self._viz_counter = 0
	def visualize(self, info, logger): # records output directly to logger
		self._viz_counter += 1
		with torch.no_grad():
			self._visualize(info, logger)
	def _visualize(self, info, logger):
		raise NotImplementedError

	def pre_epoch(self):
		self.reset_viz_counter()
		super().pre_epoch()

class Optimizable(Recordable):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.optim = None

	def record_lr(self):
		if self.optim is not None:
			if isinstance(self.optim, util.Complex_Optimizer):
				for name, optim in self.optim.items():
					name = 'z-lr-{}'.format(name)
					if name not in self.stats:
						self.stats.new(name)
					lr = optim.param_groups[0]['lr']
					self.stats.update(name, lr)
			else:
				if 'z-lr' not in self.stats:
					self.stats.new('z-lr')
				lr = self.optim.param_groups[0]['lr']
				self.stats.update('z-lr', lr)

	def pre_epoch(self):
		super().pre_epoch()
		self.record_lr()

	def set_optim(self, optim_info=None):

		if optim_info is None: # aggregate optimizers of children
			sub_optims = {}
			for name, child in self.named_children():
				if isinstance(child, Optimizable) and child.optim is not None:
					sub_optims[name] = child.optim

			assert len(sub_optims) > 0, 'no children have optimizers'

			# if len(sub_optims) == 1:
			# 	optim = next(iter(sub_optims.values()))
			# else:
			# 	optim = util.Complex_Optimizer(**sub_optims)
			optim = util.Complex_Optimizer(**sub_optims)

		else:
			optim = util.default_create_optim(self.parameters(), optim_info)

		self.optim = optim


	def optim_step(self, loss): # should only be called during training
		if self.optim is None:
			raise Exception('Optimizer not set')
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	def load_state_dict(self, state_dict):
		if self.optim is not None:
			self.optim.load_state_dict(state_dict['optim'])
		super().load_state_dict(state_dict['model'])

	def state_dict(self, *args, **kwargs):
		state_dict = {
			'model': super().state_dict(*args, **kwargs),
		}
		if self.optim is not None:
			state_dict['optim'] = self.optim.state_dict()
		return state_dict


class Schedulable(Optimizable):
	def __init__(self, *args, scheduler=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.scheduler = scheduler

	def set_scheduler(self, info=None):
		assert self.optim is not None, 'no optim to schedule'
		if info is None:
			sub_sch = {}
			for name, child in self.named_children():
				if isinstance(child, Schedulable) and child.scheduler is not None:
					sub_sch[name] = child.scheduler

			if len(sub_sch) == 0:
				sch = None
			else:

				# if len(sub_sch) == 1:
				# 	sch = next(iter(sub_sch.values()))
				# else:
				# 	sch = util.Complex_Scheduler(**sub_sch)
				sch = util.Complex_Scheduler(**sub_sch)
		else:
			sch = util.default_create_scheduler(self.optim, info)

		self.scheduler = sch

	def load_state_dict(self, state_dict):
		if self.scheduler is not None:
			self.scheduler.load_state_dict(state_dict['scheduler'])
		super().load_state_dict(state_dict)

	def state_dict(self, *args, **kwargs):
		state_dict = super().state_dict(*args, **kwargs)
		if self.scheduler is not None:
			state_dict['scheduler'] = self.scheduler.state_dict()
		return state_dict

	def schedule_step(self, val=None):
		if self.scheduler is not None:
			print('LR Scheduler stepping')
			if self.scheduler.req_loss:
				self.scheduler.step(val)
			else:
				self.scheduler.step()

	def post_epoch(self, stats):
		assert 'loss' in stats and stats['loss'].count > 0, 'no metric to check'
		self.schedule_step(stats['loss'].avg.item())
		super().post_epoch()

class Regularizable(object):
	def regularize(self, q):
		return torch.tensor(0).type_as(q)

class Trainable_Model(Optimizable, Model): # top level - must be implemented to train
	def step(self, batch):  # Override pre-processing mixins
		return self._step(batch)

	def test(self, batch):  # Override pre-processing mixins
		return self._test(batch)

	def _step(self, batch, out=None):  # Override post-processing mixins
		if out is None:
			out = util.TensorDict()
		return out

	def _test(self, batch):  # Override post-processing mixins
		return self._step(batch)  # by default do the same thing as during training

	# NOTE: never call an optimizer outside of _step (not in mixinable functions)
	# NOTE: before any call to an optimizer check with self.train_me()
	def train_me(self):
		return self.training and self.optim is not None










# Old (but not deprecated)


class Transition_Model(Model):
	def __init__(self, state_dim, ctrl_dim):
		super(Transition_Model, self).__init__(state_dim + ctrl_dim, state_dim)

		self.state_dim = state_dim
		self.ctrl_dim = ctrl_dim

		self.info = util.NS()

	def sequence(self, state0, ctrls, ret_info=False, ret_all=True):

		states = [state0]
		dynamics = util.NS(**{k: [] for k in self.info})

		for ctrl in ctrls:
			states.append(self(states[-1], ctrl))

			if ret_info:
				for k in self.info:
					dynamics[k].append(self.info[k])

		if ret_all:
			output = torch.stack(states[1:])  # T x B x S
		else:
			output = states[-1]

		if ret_info:
			return output, dynamics
		return output

	def jacobians(self, x, u):  # by finite differencing

		deltas_x = torch.eye(self.state_dim).cuda() * self.eps
		deltas_u = torch.eye(self.ctrl_dim).cuda() * self.eps

		x = x.view(1, self.state_dim)
		u = u.view(1, self.ctrl_dim)

		x_inputs = torch.cat([deltas_x + x] + [x] * (self.ctrl_dim + 1))
		u_inputs = torch.cat([u] * (self.state_dim + 1) + [deltas_u + u])

		output = self(x_inputs, u_inputs)

		Jx, n, Ju = output.narrow(0, 0, self.state_dim), \
		            output.narrow(0, self.state_dim, 1), \
		            output.narrow(0, self.state_dim + 1, self.ctrl_dim)

		return (Jx - n) / self.eps, (Ju - n) / self.eps  # returns transposed jacobians

	def forward(self, state, ctrl=None):  # identity model
		return state


class Inverse_Model(Model):
	def __init__(self, state_dim, ctrl_dim):
		super(Inverse_Model, self).__init__(state_dim * 2, ctrl_dim)

	def forward(self, state, next_state):
		raise NotImplementedError



class Agent(nn.Module):
	def __init__(self, policy, stats=None):
		super(Agent, self).__init__()
		self.stats = util.StatsMeter() if stats is None else stats

		self.policy = policy
		self.state_dim, self.action_dim = self.policy.state_dim, self.policy.action_dim

	def forward(self, x):
		return self.policy.get_action(x)

	def gen_action(self, x): # used by generator, includes optional info in dict
		return self(x), {}

	def forward(self, state):
		return self.policy.get_action(state)

	def _update_policy(self, **paths): # Must be overridden
		raise NotImplementedError

	def _format_paths(self, **paths):

		if not isinstance(paths['states'], torch.Tensor):
			paths['states'] = torch.cat(paths['states'])
		if not isinstance(paths['actions'], torch.Tensor):
			paths['actions'] = torch.cat(paths['actions'])
		if not isinstance(paths['rewards'], torch.Tensor):
			paths['rewards'] = torch.cat(paths['rewards'])

		paths['states'] = paths['states'].view(-1, self.state_dim)
		paths['actions'] = paths['actions'].view(-1, self.action_dim)
		paths['rewards'] = paths['rewards'].view(-1, 1)

		return paths

	def learn(self, **paths):
		paths = self._format_paths(**paths)
		self._update_policy(**paths)
		return paths


class Policy(Model):
	def __init__(self, state_dim, action_dim):
		super(Policy, self).__init__(state_dim, action_dim)
		self.state_dim = state_dim
		self.action_dim = action_dim

	def get_action(self, state, greedy=False):
		raise NotImplementedError

	def forward(self, state):
		return self.get_action(state)


class Baseline(Model):
	def __init__(self, state_dim, value_dim, stats=None):
		super().__init__(state_dim, value_dim)
		self.stats = stats if stats is not None else util.StatsMeter()
		self.state_dim = state_dim
		self.value_dim = value_dim

	def _update(self, states, values):
		raise NotImplementedError

	def _get_value(self, states):
		raise NotImplementedError

	def forward(self, states):
		return self._get_value(states)

	def learn(self, states, values):
		self._update(states, values)


class Env(object):
	def __init__(self, spec, ID=None):
		# assert False, 'not used anymore'
		self.env_id = ID
		self.spec = spec
		self.seed()
		self._horizon = self.spec.horizon

	@property
	def horizon(self):
		return self._horizon

	def seed(self, seed=None):
		self._seed = np.random.randint(2 ** 32) if seed is None else seed
		self._tgen = torch.manual_seed(self._seed)
		self._npgen = np.random.RandomState(self._seed)
		return self._seed

	def reset(self, init_state=None):  # returns: obs
		raise NotImplementedError

	def step(self, action):  # returns: next_obs, reward, done, info
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
			stats.update('len', step + 1)

		return stats

	def visualize_policy(self, N, policy, T=None):
		return self.evaluate_policy(N, policy, T=T, render=True)

class BatchedEnv(Env):
	def __init__(self, batch_size=1, spec=None, ID=None):
		self.batch_size = batch_size
		super(BatchedEnv, self).__init__(spec, ID)

class Exploration(object):
	def __init__(self, action_dim):
		self.dout = action_dim

	def __call__(self, pi):  # gets policy distribution
		raise Exception('not overridden')  # returns action

	def reset(self):
		pass

