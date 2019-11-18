
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

from ...framework import Model

from .dynamics import Controlled_Dynamics, Cartpole_Dynamics
from .controllers import Constant, Composer, PowerForce, CutPowerForce, SimpleLimiter
from .descriptions import Typed_Dim, Angle_Dim, Discrete_Dim, Ranged_Dim, Velocity_Dim
from .representations import SinCosRepresentation

# TODO: add reward functions on top of simulators
# TODO: sim.print_desc() - nicely format state_desc and action_desc
# TODO: improve rendering api

class Simulator(Model):
	def __init__(self, dynamics, timestep, batch_size=None, state_desc=None, action_desc=None, integration_method='dopri5'):
		super().__init__((len(action_desc) if action_desc is not None else 0), dynamics.dim)

		self.dynamics = dynamics
		self.dt = timestep
		self.time = None
		self.timestep = None
		self.integration_method = integration_method

		self.batch_size = batch_size

		self.state_desc = state_desc
		self.action_desc = action_desc

	def nicify(self):
		raise NotImplementedError

	def sample_state(self, N=None):
		if self.state_desc is not None:
			state = torch.cat([desc.sample(N) for desc in self.state_desc], -1).to(self.device)
			return self._observe(state)
		raise NotImplementedError

	def sample_action(self, N=None):
		if self.action_desc is not None:
			return torch.cat([desc.sample(N) for desc in self.action_desc], -1).to(self.device)
		raise NotImplementedError

	def _observe(self, state):
		return state

	def reset(self, state=None):

		if state is None:
			state = self.sample_state(self.batch_size)

		self.state = self._reset(state)

		self.time = 0.
		self.timestep = 0

		return self._observe(self.state)

	def _reset(self, state):
		return state

	def _apply_control(self, ctrl):
		raise NotImplementedError

	def _integrate(self, N=1):

		t = torch.tensor([self.time, self.time + N*self.dt])

		next_state = odeint(self.dynamics, self.state, t, method=self.integration_method)[-1]

		return next_state

	def step(self, ctrl=None, N=1): # take N steps using ctrl (if sim can be controlled)

		if ctrl is not None:
			self._apply_control(ctrl)

		self.state = self._integrate(N)

		return self._observe(self.state)

	def render(self, W, H, index=None):
		raise NotImplementedError

class Cartpole(Simulator):

	def __init__(self, ctrl_scale=5., timestep=0.01, batch_size=None,
	             limit=1., limit_pow=4., limit_coeff=-40., # by default in a power 36 potential for limit
	             init_dx_scale=1., init_dtheta_scale=1.,
	             # dynamics=None, state_desc=None, action_desc=None,
	             integration_method='rk4',
	             **dynamics_args):

		controller = Constant()
		actuator = controller
		
		if limit is not None:
			frc = CutPowerForce(power=limit_pow, coeff=limit_coeff)
			limiter = SimpleLimiter(frc, index=0, range=limit)
			actuator = Composer(controller, limiter)
		else:
			print('WARNING: x is not limited in cartpole')


		dynamics = Cartpole_Dynamics(cart_force=actuator, **dynamics_args)

		state_desc = [Ranged_Dim(name='x', min=-1, max=1), Angle_Dim(name='theta'),
		              Velocity_Dim(name='dx', scale=init_dx_scale), Velocity_Dim(name='dtheta', scale=init_dtheta_scale)]

		action_desc = [Ranged_Dim(name='u', min=-1, max=1)]

		super().__init__(dynamics=dynamics, timestep=timestep, batch_size=batch_size, integration_method=integration_method,
		                 state_desc=state_desc, action_desc=action_desc)

		self.controller = controller

		self._true_state_desc = state_desc
		self.ctrl_scale = ctrl_scale

	def nicify(self):
		return SinCosRepresentation(self, 1)

	def sample_state(self, N=None):  # TODO: fix for sincos

		states = torch.cat([desc.sample(N) for desc in self._true_state_desc], -1).to(self.device)

		return states

	def _reset(self, state): # TODO: fix for sincos
		self.controller.val *= 0.

		return state

	def _apply_control(self, ctrl):
		self.controller.update(self.ctrl_scale * ctrl.view(-1, 1).clamp(-1,1))

	def _observe(self, state):

		# if self.sincos:
		# 	x, theta, dx, dtheta = state.narrow(-1,0,1), state.narrow(-1,1,1), state.narrow(-1,2,1), state.narrow(-1,3,1)
		# 	return torch.cat([x, torch.sin(theta), torch.cos(theta), dx, dtheta],-1)

		state[...,1] %= 2*np.pi

		return super()._observe(state)

	def _render(self, state, size):
		W, H = size
		assert W > 10 and H > 10, 'too small'

		img = Image.new("RGB", (W, H), (0, 0, 0, 0))

		dr = ImageDraw.Draw(img)

		level = H / 2
		space = W / 8

		unit = 3 * W / 8
		L = unit * self.dynamics.length

		# state = self.state if index is None else self.state[index]
		state = state.cpu().numpy()

		# rail
		dr.line([space, level, W - space, level], width=4, fill=(150, 100, 0))

		cart = state[0] * unit + W / 2

		# pole
		theta = state[1]
		# px, py = state[1:3] if self.sincos else (np.sin(theta), np.cos(theta))
		px, py = np.sin(theta), np.cos(theta)

		dr.line([cart, level, cart + L * px, level - L * py], width=5, fill=(80, 100, 220))

		# cart
		cW, cH = max(W / 16, 1), max(H / 16, 1)
		dr.rectangle([cart - cW, level - cH, cart + cW, level + cH], fill=(180, 50, 20))

		del dr
		return np.array(img)

	def render(self, W, H, index=None):

		if self.state.ndimension() == 2:
			if index is None:
				return np.stack([self._render(s, size=(W,H)) for s in self.state])
			return self._render(self.state[index], size=(W,H))
		return self._render(self.state, size=(W,H))












