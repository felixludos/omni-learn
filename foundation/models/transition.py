import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import numpy as np
import ctrlnets
import dynnets
import util
from util.misc import NS
from .. import framework as fm

class Periodic_Transition_Model(fm.Transition_Model): # can have periodic dims (treated as cossin)
	def __init__(self, state_dim, ctrl_dim, periodic=None):
		super(Periodic_Transition_Model, self).__init__(state_dim, ctrl_dim)

		if periodic is None:
			self.periodic = np.zeros(self.state_dim, dtype=bool)
		elif isinstance(periodic[0], bool):
			self.periodic = np.arange(self.state_dim)[periodic]
		elif isinstance(periodic[0], int):
			self.periodic = periodic
		else:
			raise Exception('unknown periodic dims choice: {}'.format(periodic))

		self.is_periodic = len(self.periodic) > 0

class PV_Transition_Model(Periodic_Transition_Model): # pos/vel structure (finite diff pos to get vel)
	def __init__(self, state_dim, ctrl_dim, vel_dim=None, periodic=None, dt=1.):
		super(PV_Transition_Model, self).__init__(state_dim, ctrl_dim, periodic)

		self.vel_dim = vel_dim
		self.pos_dim = self.state_dim - self.vel_dim

		self.dt = dt

####################################
### Transition models (predicts change in poses based on the applied control)

class NaiveTransitionModel(PV_Transition_Model):
	def __init__(self, nonlinearity='prelu', hidden_dims=[48, 64, 64], **kwargs):
		super(NaiveTransitionModel, self).__init__(**kwargs)

		self.net = util.make_MLP(input_dim=self.state_dim + self.ctrl_dim,
									 output_dim=self.state_dim,
									 hidden_dims=hidden_dims,
									 nonlinearity=nonlinearity)

	def forward(self, state, ctrl):
		# Run the forward pass

		s = state.view(-1, self.state_dim)
		u = ctrl.view(-1, self.ctrl_dim)

		ns = self.net(torch.cat([s, u], 1))  # Encode pose+ctrl

		if self.is_periodic: # constrain cossin
			ns[:, self.periodic] = F.tanh(ns[:, self.periodic])

		return ns  # next state


class DeltaTransitionModel(PV_Transition_Model):
	def __init__(self, nonlinearity='prelu', hidden_dims=[48, 64, 64], **kwargs):
		super(DeltaTransitionModel, self).__init__(**kwargs)

		#assert self.finite_diff, 'Deltas cannot be predicted when composing (yet)'

		self.net = util.make_MLP(input_dim=self.state_dim + self.ctrl_dim,
									 output_dim=self.state_dim,
									 hidden_dims=hidden_dims,
									 nonlinearity=nonlinearity)

		# save jacobians to self.dynamics

	def forward(self, state, ctrl):
		# Run the forward pass

		s = state.view(-1, self.state_dim)
		u = ctrl.view(-1, self.ctrl_dim)

		ds = self.net(torch.cat([s, u], 1))  # Encode pose+ctrl

		return s + ds*self.dt


class PosTransitionModel(PV_Transition_Model):
	def __init__(self, nonlinearity='prelu', hidden_dims=[48, 64, 64], offset_vel=True, **kwargs):
		super(PosTransitionModel, self).__init__(**kwargs)

		self.offset_vel = offset_vel

		dout = self.pos_dim + (self.vel_dim if offset_vel else 0)
		self.net = util.make_MLP(input_dim=self.state_dim + self.ctrl_dim,
									 output_dim=dout,
									 hidden_dims=hidden_dims,
									 nonlinearity=nonlinearity)

	def forward(self, state, ctrl):
		# Run the forward pass

		s = state.view(-1, self.state_dim)
		u = ctrl.view(-1, self.ctrl_dim)

		y = self.net(torch.cat([s, u], 1))  # Encode pose+ctrl

		if self.offset_vel:
			nx, o = y.narrow(1, 0, self.pos_dim), y.narrow(1, self.pos_dim, self.vel_dim)
		else:
			nx, o = y, 0

		if self.is_periodic: # constrain cossin
			nx[:, self.periodic] = F.tanh(nx[:, self.periodic])

		x = s[:, :self.pos_dim]

		nv = (nx - x + o) / self.dt

		return torch.cat([nx, nv],-1).view(-1, self.state_dim) # next state


class AccTransitionModel(PV_Transition_Model):
	def __init__(self, nonlinearity='prelu', hidden_dims=[48, 64, 64], offset_pos=True, **kwargs):
		super(AccTransitionModel, self).__init__(**kwargs)

		self.offset_pos = offset_pos

		dout = self.vel_dim + (self.pos_dim if offset_pos else 0)
		self.net = util.make_MLP(input_dim=self.state_dim + self.ctrl_dim,
									 output_dim=dout,
									 hidden_dims=hidden_dims,
									 nonlinearity=nonlinearity)

	def forward(self, state, ctrl):
		# Run the forward pass

		s = state.view(-1, self.state_dim)
		u = ctrl.view(-1, self.ctrl_dim)

		y = self.net(torch.cat([s, u], 1))  # Encode pose+ctrl

		if self.offset_pos:
			a, o = y.narrow(1, 0, self.vel_dim), y.narrow(1, self.vel_dim, self.pos_dim)
		else:
			a, o = y, 0

		x, v = s[:, :self.pos_dim], s[:, -self.vel_dim:]

		nv = self.dt*a + v
		nx = self.dt*nv + x + o

		return torch.cat([nx, nv],-1).view(-1, self.state_dim) # next state


class LinearTransitionModel(PV_Transition_Model):
	def __init__(self, nonlinearity='prelu', hidden_dims=[48, 64, 64], **kwargs):
		super(LinearTransitionModel, self).__init__(**kwargs)

		self.net = util.make_MLP(input_dim=self.state_dim,
									 output_dim=self.pos_dim + (1 + self.ctrl_dim) * self.vel_dim,
									 hidden_dims=hidden_dims,
									 nonlinearity=nonlinearity)

	def forward(self, state, ctrl): # Bx2xNx3, BxC
		# Run the forward pass

		s = state.view(-1,self.state_dim)
		y = self.net(s)  # Encode se3_typepose+ctrl

		ox, ov, B = y.narrow(1,0,self.pos_dim), y.narrow(1, self.pos_dim, self.vel_dim), y[:,-self.vel_dim*self.ctrl_dim:]
		# BxSx1, BxSx1, BxS*Cx1,

		B = B.contiguous().view(-1, self.vel_dim, self.ctrl_dim)

		u = ctrl.unsqueeze(-1)  # BxCx1

		x, v = s[:,self.pos_dim:], s[:,-self.vel_dim:] # BxNx3, BxNx3

		a = B.bmm(u).view(-1, self.vel_dim)

		# integration with offsets
		nv = ov + self.dt*a + v
		nx = self.dt*nv + x + ox

		return torch.cat([nx, nv],-1).view(-1, self.state_dim)



