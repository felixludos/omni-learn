
import torch
from torch import nn

from .descriptions import Ranged_Dim, SinCos_Dim

class Representation(object):

	def __init__(self, sim):
		self.sim = sim

		self.state_desc = self.fix_desc(self.sim.state_desc)

	def fix_desc(self, state_desc):
		raise NotImplementedError

	def encode(self, x): # user state -> internal state
		raise NotImplementedError

	def decode(self, q): # internal state -> user state
		raise NotImplementedError

	def reset(self, state=None):
		if state is not None:
			state = self.encode(state)
		return self.decode(self.sim.reset(state))

	def step(self, ctrl=None, N=1):
		return self.decode(self.sim.step(ctrl=ctrl, N=N))

	def sample_state(self, N=1):
		return self.decode(self.sim.sample_state(N=N))

	def __getattribute__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return getattr(self.sim, item)

class SinCosRepresentation(Representation):

	def __init__(self, sim, index):

		try:
			len(index)
		except TypeError:
			index = [index]

		self.index = index

		super().__init__(sim)

	def fix_desc(self, state_desc):

		self.old_desc = state_desc
		self.state_desc = []
		self.other = []

		for i, desc in enumerate(state_desc):
			if i in self.index:
				continue

			self.other.append(i)
			self.state_desc.append(desc)

		for i in self.index:
			desc = state_desc[i]
			self.state_desc.append(SinCos_Dim(name=desc.name, type='sin'))

		for i in self.index:
			desc = state_desc[i]
			self.state_desc.append(SinCos_Dim(name=desc.name, type='cos'))

		return self.state_desc

	def encode(self, x):

		sqz = False
		if x.ndimension() == 1:
			sqz = True
			x = x.unsqueeze(0)

		nonangles, sincos = x[..., :len(self.other)], x[..., len(self.other):]

		sin, cos = sincos[...,:len(self.index)], sincos[...,len(self.index):]

		angles = torch.atan2(cos, sin)

		q = torch.zeros((angles.size(0), len(self.old_desc)), device=x.device)

		q[:,self.index] += angles
		q[:,self.other] += nonangles

		if sqz:
			q = q.squeeze(0)

		return q

	def decode(self, q):

		angles, nonangles = q[...,self.index], q[...,self.other]

		sin, cos = torch.sin(angles), torch.cos(angles)

		return torch.cat([nonangles, sin, cos], -1)



