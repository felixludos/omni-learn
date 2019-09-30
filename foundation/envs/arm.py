import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from . import general as gen
from .. import util
from .. import framework as fm
from ..sim import integrators
import matplotlib.pyplot as plt


class Arm(fm.Env):  # Touching the target
	def __init__(self, num_links=2, sparse_reward=True, render_target=False, moving_target=False, link_lens=None,
				 link_masses=None, cossin=False, include_pos=True, alpha=None, beta=None, gamma=None, ):
		self.cossin = cossin
		self.spec = gen.EnvSpec(obs_space=gen.Continuous_Space(shape=(num_links * (3 if self.cossin else 2) + 4)),
		                        act_space=gen.Continuous_Space(shape=(num_links)),
		                        horizon=50)
		super(Arm, self).__init__(self.spec)

		assert not render_target
		assert not moving_target
		assert not cossin
		assert include_pos
		self.obs_size = 4 + (2 if include_pos else 0)
		self.include_pos = include_pos

		assert num_links == 2, 'only set up for double pendulum'

		# these params stay constant for all episodes
		self.N = num_links
		assert link_lens is None or len(link_lens) == num_links
		assert link_masses is None or len(link_masses) == num_links

		self.set_lengths(link_lens)
		self.set_masses(link_masses)

		self.sparse_rewards = sparse_reward
		self.moving_target = moving_target
		self.render_target = render_target

		self.obs = None
		self.cmd = None

		dynamics = self._double_pendulum_dynamics if self.N == 2 else self._simple_dynamics

		self.sim = integrators.Velocity_Verlet(dynamics, None, None, 1. / 720,
											   coord_jacobian=self._coord_jacobian if self.moving_target else None)  # control at 60 hz
		# self.sim = integrators.Velocity_Verlet(dynamics, None, None, 1. / 720,)

		self.set_ctrl_freq(60.)

		self.alpha = torch.FloatTensor([4.,2.5]) if alpha is None else alpha #torch.FloatTensor([4, 2])  # ctrl scale ( gear)
		self.gamma = torch.ones(2)*0.1 if gamma is None else gamma  # damping
		self.beta = torch.FloatTensor([5.])  if beta is None else beta # gravity

		self.vel_std = 1
		self.ctrl_lim = 1

		self._render_size = (128, 128)
		self._render_hand_kwargs = {'color': '#1f77b4', 'marker': 'o', 'ms': 6}  # blue: '#1f77b4', orange: '#ff7f0e'
		self._render_goal_kwargs = {'color': '#2ca02c', 'marker': 'o', 'ms': 6}
		self._render_arm_kwargs = {'color': 'k', 'ls': '-', 'lw': 2.5, 'marker': 'o', 'ms': 4}
		self._render_base_kwargs = {'color': 'k', 'marker': 'o', 'ms': 7}

	def set_masses(self, masses=None):
		if masses is None:
			masses = torch.ones(self.N)
		self.masses = masses

	def set_lengths(self, lengths=None):
		if lengths is None:
			lengths = torch.ones(self.N)
		self.lengths = lengths.div(lengths.sum())

	def _compute_inertia(self):
		self.inertia = self.masses * self.lengths ** 2 / 3.

	def set_ctrl_freq(self, freq):  # in hz
		self.n_steps = int(np.round(1. / freq / self.sim.timestep))
		self.ctrl_freq = 1. / (self.n_steps * self.sim.timestep)

	def _coord_jacobian(self, x):
		j = torch.ones(x.size(0))
		j[self.N + 1] = x[self.N]  # dx,dy = r*dtheta
		return j

	def _simple_dynamics(self, t, x, dx):  # simple dynamics - no interaction between segments of pendulum
		return (self.cmd - self.gamma * dx - self.beta * torch.cos(x)) / self.inertia

	def _double_pendulum_dynamics(self, t, x, dx):
		ddx = torch.zeros(x.size(0))
		m = self.masses
		l = self.lengths
		g = self.beta
		sin = torch.sin
		cos = torch.cos

		a1 = (l[1] / l[0]) * (m[1] / m.sum()) * cos(x[0] - x[1])
		a2 = (l[0] / l[1]) * cos(x[0] - x[1])

		f1 = -(l[1] / l[0]) * (m[1] / m.sum()) * dx[1] ** 2 * sin(x[0] - x[1]) - g / l[0] * cos(x[0])
		f2 = (l[0] / l[1]) * dx[0] ** 2 * sin(x[0] - x[1]) - g / l[1] * cos(x[1])

		ddx[0] = f1 - a1 * f2
		ddx[1] = -a2 * f1 + f2

		ddx[:2] /= 1 - a1 * a2

		ddx[:2] -= (self.gamma * dx[:2] - self.alpha * self.cmd) / self.inertia

		if self.moving_target:
			ddx = self._target_dynamics(ddx, t, x, dx)

		return ddx

	def _complex_weird_dynamics(self, t, x, dx):  # seems stable but dont use
		ddx = torch.zeros(self.N)
		m = self.masses
		l = self.lengths
		g = self.beta
		sin = torch.sin
		cos = torch.cos
		tan = torch.tan
		ddx[0] = g * ((m[0] + m[1]) * cos(x[0]) - m[1] * sin(x[0]) / tan(x[1])) / l[0] / m[0]
		ddx[1] = 1. / (2 * l[1] * m[0] * sin(x[1])) * (
					-g * (2 * m[0] + m[1] - m[1] * cos(2 * x[0])) / tan(x[1]) + g * (m[0] + m[1]) * sin(2 * x[0]) + 2 *
					l[0] * m[0] * cos(x[0]) * dx[0] ** 2 + 2 * l[1] * m[0] * cos(x[1]) * dx[1] ** 2)
		return ddx

	def _target_dynamics(self, ddx, t, x, dx):
		if x[self.N] >= 0.96 and dx[self.N] > 0:  # goal collision w/ boundary - impulse
			ddx[self.N] = - 2 * dx[self.N] / self.sim.timestep
		if x[self.N] <= 0.04 and dx[self.N] < 0:
			ddx[self.N] = - 2 * dx[self.N] / self.sim.timestep
		return ddx

	def _make_obs(self, x, dx):
		if self.obs is None:
			self.obs = torch.zeros(self.obs_size)

		cs, ss = torch.cos(x), torch.sin(x)

		self.obs[:self.N] = x % (2*np.pi)
		self.obs[self.N:self.N * 2] = dx
		if self.include_pos:
			self.obs[self.N * 2] = cs.mul(self.lengths).sum()
			self.obs[self.N * 2 + 1] = ss.mul(self.lengths).sum()

		return self.obs.clone()

	def _compute_reward(self, obs, action):
		reward = -(obs[-4:-2] - obs[-2:]).norm().item()
		if self.sparse_rewards:
			reward = float(reward > -0.05)
		return reward

	def reset(self, x_0=None, dx_0=None, goal=None):

		if dx_0 is None and x_0 is not None and x_0.size(0) >= 2*self.N:
			dx_0 = x_0[self.N:self.N*2]
			x_0 = x_0[:self.N]
		elif dx_0 is None:
			dx_0 = torch.randn(self.N + 2) * self.vel_std
			if self.moving_target:
				dx_0[self.N:self.N + 2] = torch.randn(2) / 2 + 0.3
				dx_0[self.N:self.N + 2][torch.rand(2) < 0.5] *= -1
				dx_0[self.N + 1] *= 2
		if x_0 is None:
			x_0 = torch.rand(self.N + 2) * 2 * np.pi
			if self.moving_target:
				x_0[self.N] = np.sqrt(np.random.rand())
				x_0[self.N + 1] = np.random.rand() * 2 * np.pi
		if goal is None:
			if self.moving_target:
				goal = torch.zeros(2)
				goal[0] = x_0[self.N] * torch.cos(x_0[self.N + 1])
				goal[1] = x_0[self.N] * torch.sin(x_0[self.N + 1])
			else:
				goal = torch.randn(2)
				goal /= goal.norm()  # radius = 1
				goal *= np.sqrt(np.random.rand())  # random radius from center

		self.goal = goal

		if not self.moving_target:
			x_0 = x_0[:self.N]
			dx_0 = dx_0[:self.N]

		self.cmd = 0
		self._compute_inertia()

		self.sim.reset(x_0, dx_0)

		return self._make_obs(x_0[:self.N], dx_0[:self.N])

	def step(self, action):

		self.cmd = action[:self.N].clamp(-self.ctrl_lim, self.ctrl_lim)
		x, dx = self.sim.step(self.n_steps)

		if self.moving_target:
			self.goal[0] = x[self.N] * torch.cos(x[self.N + 1])
			self.goal[1] = x[self.N] * torch.sin(x[self.N + 1])
			x = x[:self.N]
			dx = dx[:self.N]

		obs = self._make_obs(x, dx)

		reward = 0. #self._compute_reward(obs, action)

		return obs, reward, False, {}

	def render(self, mode='rgb', figax=None, label=False, img_shape=None):

		assert self.obs is not None, 'must reset env first'

		if img_shape is None:
			img_shape = self._render_size
		if figax is None:
			figax = plt.subplots(figsize=(img_shape[0] / 100, img_shape[1] / 100))

		fig, ax = figax
		ax.set_xlim(-1.05, 1.05)
		ax.set_ylim(-1.05, 1.05)

		if self.render_target:
			ax.plot([self.goal[0].item()], [self.goal[1].item()], **self._render_goal_kwargs)

		x, y = [0] + torch.cos(self.obs[:self.N]).mul(self.lengths).cumsum(0).numpy().tolist(), \
			   [0] + torch.sin(self.obs[:self.N]).mul(self.lengths).cumsum(0).numpy().tolist()

		ax.plot(x, y, **self._render_arm_kwargs)
		ax.plot([0], [0], **self._render_base_kwargs)
		ax.plot(x[-1:], y[-1:], **self._render_hand_kwargs)
		ax.add_artist(plt.Circle((0, 0), 1, color='r', lw=2, fill=False))  # boundary

		if mode == 'onscreen':
			return

		fig.subplots_adjust(bottom=0., left=0., right=1., top=1.)  # full sized image
		rgb = util.fig_to_rgba(fig)[:, :, :-1]

		plt.close(fig)

		if label:

			assert self.N == 2  # only supported for double pendulum

			lbl_color = lambda n: '#{}'.format("{0:#0{1}x}".format(n, 8)[2:])
			n = 0

			colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
			lbl_color = lambda n: colors[n]

			fig, ax = plt.subplots(figsize=(img_shape[0] / 100, img_shape[1] / 100))
			ax.set_xlim(-1.05, 1.05)
			ax.set_ylim(-1.05, 1.05)
			ax.axis('off')

			bg = lbl_color(n)  # 0
			n += 1
			ax.add_artist(plt.Circle((0, 0), 2, color=bg, fill=True))  # background

			if self.render_target:
				goal_lbl_args = self._render_goal_kwargs.copy()
				goal_lbl_args['color'] = lbl_color(n)
				n += 1

				ax.plot([self.goal[0].item()], [self.goal[1].item()], **goal_lbl_args)

			link_colors = [lbl_color(i) for i in range(n, n + self.N)]
			n += self.N

			arm_lbl_args = self._render_arm_kwargs.copy()
			mk = arm_lbl_args['marker']
			arm_lbl_args['marker'] = ''

			for i, (ex, ey, sx, sy, c) in enumerate(zip(x, y, x[1:], y[1:], link_colors)):
				arm_lbl_args['color'] = c
				ax.plot([sx, ex], [sy, ey], **arm_lbl_args)
				if i == 0:
					base_lbl_args = self._render_base_kwargs.copy()
					base_lbl_args['color'] = bg
					ax.plot([0], [0], **self._render_base_kwargs)

			arm_lbl_args['ls'] = ''
			arm_lbl_args['marker'] = mk

			for sx, sy, c in zip(x[1:], y[1:], link_colors):
				arm_lbl_args['color'] = c
				ax.plot([sx], [sy], **arm_lbl_args)

			hand_lbl_args = self._render_hand_kwargs.copy()
			hand_lbl_args['color'] = arm_lbl_args['color']  # color of last
			ax.plot(x[-1:], y[-1:], **hand_lbl_args)

			fig.subplots_adjust(bottom=0., left=0., right=1., top=1.)  # full sized image
			lbl = util.fig_to_rgba(fig)[:, :, :-1]
			bg = lbl.sum(2) == 0
			lbl = lbl.argmax(2) + 1
			lbl[bg] = 0

			plt.close(fig)

			return rgb, lbl

		return rgb

class OldArm(fm.Env): # Touching the target
	def __init__(self, num_links=2, sparse_reward=True, render_target=False, moving_target=False, link_lens=None, link_masses=None, cossin=True):
		self.cossin = cossin
		self.spec = gen.EnvSpec(obs_space=gen.Continuous_Space(shape=(num_links*(3 if self.cossin else 2)+4)),
		                        act_space=gen.Continuous_Space(shape=(num_links)),
		                        horizon=50)
		super(Arm, self).__init__(self.spec)

		assert num_links == 2, 'only set up for double pendulum'
		
		# these params stay constant for all episodes
		self.N = num_links
		assert link_lens is None or len(link_lens) == num_links
		assert link_masses is None or len(link_masses) == num_links

		self.set_lengths(link_lens)
		self.set_masses(link_masses)

		self.sparse_rewards = sparse_reward
		self.moving_target = moving_target
		self.render_target = render_target
		
		self.obs = None
		self.cmd = None

		dynamics = self._double_pendulum_dynamics if self.N == 2 else self._simple_dynamics

		self.sim = integrators.Velocity_Verlet(dynamics, None, None, 1./720,
											   coord_jacobian=self._coord_jacobian if self.moving_target else None) # control at 60 hz
		# self.sim = integrators.Velocity_Verlet(dynamics, None, None, 1. / 720,)
		
		self.set_ctrl_freq(60.)

		self.alpha = 5. # ctrl scale
		self.gamma = 0.1 # damping
		self.beta = 5. # gravity
		self.vel_std = 1
		self.ctrl_lim = 1

		self._render_size = (128, 128)
		self._render_hand_kwargs = {'color': '#1f77b4', 'marker':'o', 'ms':6} # blue: '#1f77b4', orange: '#ff7f0e'
		self._render_goal_kwargs = {'color':'#2ca02c', 'marker':'o', 'ms':6}
		self._render_arm_kwargs = {'color':'k', 'ls':'-', 'lw':2.5, 'marker':'o', 'ms':4}
		self._render_base_kwargs = {'color':'k', 'marker':'o', 'ms':7}

	def set_masses(self, masses=None):
		if masses is None:
			masses = torch.ones(self.N)
		self.masses = masses

	def set_lengths(self, lengths=None):
		if lengths is None:
			lengths = torch.ones(self.N)
		self.lengths = lengths.div(lengths.sum())

	def _compute_inertia(self):
		self.inertia = self.masses * self.lengths ** 2 / 3.

	def set_ctrl_freq(self, freq): # in hz
		self.n_steps = int(np.round(1./freq/self.sim.timestep))
		self.ctrl_freq = 1. / (self.n_steps * self.sim.timestep)

	def _coord_jacobian(self, x):
		j = torch.ones(x.size(0))
		j[self.N+1] = x[self.N] # dx,dy = r*dtheta
		return j

	def _simple_dynamics(self, t, x, dx): # simple dynamics - no interaction between segments of pendulum
		return (self.cmd - self.gamma * dx - self.beta * torch.cos(x)) / self.inertia

	def _double_pendulum_dynamics(self, t, x, dx):
		ddx = torch.zeros(x.size(0))
		m = self.masses
		l = self.lengths
		g = self.beta
		sin = torch.sin
		cos = torch.cos

		a1 = (l[1] / l[0]) * (m[1] / m.sum()) * cos(x[0] - x[1])
		a2 = (l[0] / l[1]) * cos(x[0] - x[1])

		f1 = -(l[1] / l[0]) * (m[1] / m.sum()) * dx[1] ** 2 * sin(x[0] - x[1]) - g / l[0] * cos(x[0])
		f2 = (l[0] / l[1]) * dx[0] ** 2 * sin(x[0] - x[1]) - g / l[1] * cos(x[1])

		ddx[0] = f1 - a1*f2
		ddx[1] = -a2*f1 + f2

		ddx[:2] /= 1 - a1*a2

		ddx[:2] -= (dx[:2] * self.gamma + self.cmd) / self.inertia

		if self.moving_target:
			ddx = self._target_dynamics(ddx, t, x, dx)

		return ddx

	def _complex_weird_dynamics(self, t, x, dx): # seems stable but dont use
		ddx = torch.zeros(self.N)
		m = self.masses
		l = self.lengths
		g = self.beta
		sin = torch.sin
		cos = torch.cos
		tan = torch.tan
		ddx[0] = g*((m[0]+m[1])*cos(x[0]) - m[1]*sin(x[0])/tan(x[1]))/l[0]/m[0]
		ddx[1] = 1./(2*l[1]*m[0]*sin(x[1]))*(-g*(2*m[0]+m[1]-m[1]*cos(2*x[0]))/tan(x[1]) + g*(m[0]+m[1])*sin(2*x[0]) + 2*l[0]*m[0]*cos(x[0])*dx[0]**2 + 2*l[1]*m[0]*cos(x[1])*dx[1]**2)
		return ddx

	def _target_dynamics(self, ddx, t, x, dx):
		if x[self.N] >= 0.96 and dx[self.N] > 0: # goal collision w/ boundary - impulse
			ddx[self.N] = - 2 * dx[self.N] / self.sim.timestep
		if x[self.N] <= 0.04 and dx[self.N] < 0:
			ddx[self.N] = - 2 * dx[self.N] / self.sim.timestep
		return ddx

	def _make_obs(self, x, dx):
		if self.obs is None:
			self.obs = torch.zeros(self.spec.obs_space.size)
			
		cs, ss = torch.cos(x), torch.sin(x)
		
		self.obs[:self.N] = cs
		self.obs[self.N:self.N*2] = ss
		self.obs[self.N*2:self.N*3] = dx
		self.obs[self.N*3] = self.goal[0]
		self.obs[self.N*3+1] = self.goal[1]
		self.obs[self.N*3+2] = cs.mul(self.lengths).sum()
		self.obs[self.N*3+3] = ss.mul(self.lengths).sum()
		
		return self.obs
	
	def _compute_reward(self, obs, action):
		reward = -(obs[-4:-2] - obs[-2:]).norm().item()
		if self.sparse_rewards:
			reward = float(reward > -0.05)
		return reward
		
	def reset(self, x_0=None, dx_0=None, goal=None):

		if x_0 is None:
			x_0 = torch.rand(self.N + 2) * 2 * np.pi
			if self.moving_target:
				x_0[self.N] = np.sqrt(np.random.rand())
				x_0[self.N + 1] = np.random.rand()*2*np.pi
		if dx_0 is None:
			dx_0 = torch.randn(self.N + 2) * self.vel_std
			if self.moving_target:
				dx_0[self.N:self.N + 2] = torch.randn(2)/2 + 0.3
				dx_0[self.N:self.N + 2][torch.rand(2) < 0.5] *= -1
				dx_0[self.N + 1] *= 2
		if goal is None:
			if self.moving_target:
				goal = torch.zeros(2)
				goal[0] = x_0[self.N] * torch.cos(x_0[self.N + 1])
				goal[1] = x_0[self.N] * torch.sin(x_0[self.N + 1])
			else:
				goal = torch.randn(2)
				goal /= goal.norm()  # radius = 1
				goal *= np.sqrt(np.random.rand())  # random radius from center

		self.goal = goal

		if not self.moving_target:
			x_0 = x_0[:self.N]
			dx_0 = dx_0[:self.N]

		self.cmd = 0
		self._compute_inertia()
		self.sim.reset(x_0, dx_0)
		
		return self._make_obs(x_0[:self.N], dx_0[:self.N])
		
	def step(self, action):
		
		self.cmd = self.alpha * action[:self.N].clamp(-self.ctrl_lim,self.ctrl_lim)
		x, dx = self.sim.step(self.n_steps)

		if self.moving_target:
			self.goal[0] = x[self.N] * torch.cos(x[self.N + 1])
			self.goal[1] = x[self.N] * torch.sin(x[self.N + 1])
			x = x[:self.N]
			dx = dx[:self.N]

		obs = self._make_obs(x, dx)
		
		reward = self._compute_reward(obs, action)
		
		return obs, reward, False, {}
	
	def render(self, mode='rgb', figax=None, label=False, img_shape=None):
		
		assert self.obs is not None, 'must reset env first'

		if img_shape is None:
			img_shape = self._render_size
		if figax is None:
			figax = plt.subplots(figsize=(img_shape[0]/100,img_shape[1]/100))
			
		fig, ax = figax
		ax.set_xlim(-1.05,1.05)
		ax.set_ylim(-1.05,1.05)

		if self.render_target:
			ax.plot([self.goal[0].item()], [self.goal[1].item()], **self._render_goal_kwargs)
		
		x, y = [0]+self.obs[:self.N].mul(self.lengths).cumsum(0).numpy().tolist(), \
			   [0] +self.obs[self.N:2*self.N].mul(self.lengths).cumsum(0).numpy().tolist()
		
		ax.plot(x, y, **self._render_arm_kwargs)
		ax.plot([0], [0], **self._render_base_kwargs)
		ax.plot(x[-1:], y[-1:], **self._render_hand_kwargs)
		ax.add_artist(plt.Circle((0, 0), 1, color='r', lw=2, fill=False)) # boundary

		if mode == 'onscreen':
			return

		fig.subplots_adjust(bottom=0., left=0., right=1., top=1.) # full sized image
		rgb = util.fig_to_rgba(fig)[:,:,:-1]

		plt.close(fig)

		if label:

			assert self.N == 2 # only supported for double pendulum

			lbl_color = lambda n: '#{}'.format("{0:#0{1}x}".format(n,8)[2:])
			n = 0

			colors = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
			lbl_color = lambda n: colors[n]

			fig, ax = plt.subplots(figsize=(img_shape[0]/100,img_shape[1]/100))
			ax.set_xlim(-1.05, 1.05)
			ax.set_ylim(-1.05, 1.05)
			ax.axis('off')

			bg = lbl_color(n)  # 0
			n += 1
			ax.add_artist(plt.Circle((0, 0), 2, color=bg, fill=True))  # background

			if self.render_target:

				goal_lbl_args = self._render_goal_kwargs.copy()
				goal_lbl_args['color'] = lbl_color(n)
				n+=1

				ax.plot([self.goal[0].item()], [self.goal[1].item()], **goal_lbl_args)

			link_colors = [lbl_color(i) for i in range(n,n+self.N)]
			n += self.N

			arm_lbl_args = self._render_arm_kwargs.copy()
			mk = arm_lbl_args['marker']
			arm_lbl_args['marker'] = ''

			for i, (ex, ey, sx, sy, c) in enumerate(zip(x,y,x[1:], y[1:], link_colors)):
				arm_lbl_args['color'] = c
				ax.plot([sx,ex], [sy,ey], **arm_lbl_args)
				if i == 0:
					base_lbl_args = self._render_base_kwargs.copy()
					base_lbl_args['color'] = bg
					ax.plot([0], [0], **self._render_base_kwargs)

			arm_lbl_args['ls'] = ''
			arm_lbl_args['marker'] = mk

			for sx, sy, c in zip(x[1:], y[1:], link_colors):
				arm_lbl_args['color'] = c
				ax.plot([sx], [sy], **arm_lbl_args)

			hand_lbl_args = self._render_hand_kwargs.copy()
			hand_lbl_args['color'] = arm_lbl_args['color'] # color of last
			ax.plot(x[-1:], y[-1:], **hand_lbl_args)

			fig.subplots_adjust(bottom=0., left=0., right=1., top=1.)  # full sized image
			lbl = util.fig_to_rgba(fig)[:,:,:-1]
			bg = lbl.sum(2) == 0
			lbl = lbl.argmax(2) + 1
			lbl[bg] = 0

			plt.close(fig)

			return rgb, lbl

		return rgb


#def render_arm(state, )




