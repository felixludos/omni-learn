import numpy as np
import torch

# control scheme
class MPC(object):

	def __init__(self, controller, horizon, state_dim, action_dim, warm_start=True, init_type='zero'): # called before all episodes
		self.controller = controller
		self.horizon = horizon
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.warm_start = warm_start
		self.init_type = init_type

	def reset(self): # called at the beginning of each episode
		if self.init_type == "random":
			u_init = torch.rand(self.horizon, self.action_dim).cuda() * 6 - 3
		else:
			u_init = torch.zeros(self.horizon, self.action_dim).cuda()

		self.controller.reset(u_init)

	def eval_trajectory(self, x_traj, u_seq): # called at the end of each episode
		raise NotImplementedError

	def get_next_command(self, state, step): # called each timestep - optimize as necessary and apply policy
		raise NotImplementedError

class OnlineMPC(MPC): # TODO: allow for optimization over parts of episode (not just every step)
	def __init__(self, controller, horizon, state_dim, action_dim, warm_start=True, init_type='zero', ep_len=None): # set ep_len for receding horizon
		super(OnlineMPC, self).__init__(controller, horizon, state_dim, action_dim, warm_start,init_type)
		self._total_len = ep_len

		assert ep_len is None or ep_len > self.horizon
		assert init_type == 'zero'

	def reset(self):
		super(OnlineMPC, self).reset()
		self._step_counter = -1

	def eval_trajectory(self, x_traj, u_seq):
		# for online mpc - eval traj just evaluates the cost of the trajectory
		return self.controller.eval_trajectory_cost(x_traj, u_seq)

	def get_next_command(self, state, step):
		delta_step = step - self._step_counter
		if not self.warm_start:
			self.controller._u_opt *= 0.
		elif self.controller._u_opt is not None: # warm start if possible
			self.controller._u_opt[:-delta_step] = self.controller._u_opt[delta_step:]
			self.controller._u_opt[-delta_step:] = 0.

		seq_len = self.horizon if self._total_len is None else min(self.horizon, self._total_len - self._step_counter)
		self.controller.optimize(state, seq_len)

		self._step_counter += delta_step
		return self.controller(state, step)

class IterativeMPC(MPC):
	def __init__(self, *args, **kwargs):
		super(IterativeMPC, self).__init__(*args, **kwargs)

		# init now before all episodes
		super(IterativeMPC, self).reset()

	def reset(self): # keep params from previous episodes
		if not self.warm_start:
			super(IterativeMPC, self).reset()

	def eval_trajectory(self, x_traj, u_seq):
		# for iterative mpc - eval traj updates policy based on traj
		return self.controller.eval_policy(x_traj, u_seq)

	def get_next_command(self, state, step):
		if step == 0:
			self.controller.optimize(state, self.horizon)
		return self.controller(state, step)
