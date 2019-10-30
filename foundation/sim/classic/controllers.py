
import torch
from torch import nn
from torch.nn import functional as F


class Controller(nn.Module): # force controller

	def __init__(self, dim):
		super().__init__()

		self.dim = dim

	def forward(self, t, q):
		raise NotImplementedError # returns [_, dim] forces


class Composer(Controller):
	def __init__(self, *controllers):
		super().__init__(controllers[0].dim)

		for ctrl in controllers:
			if self.dim != ctrl.dim:
				raise Exception('not all controllers have the same dimensionality')

		self.controllers = nn.ModuleList(controllers)

	def __getitem__(self, item):
		return self.controllers[item]

	def forward(self, t, q):
		U = self.controllers[0](t, q)
		for ctrl in self.controllers[1:]:
			U += ctrl(t, q)
		return U


class Constant(Controller):
	def __init__(self, val=None):

		if val is None:
			val = torch.zeros(1).unsqueeze(0)

		super().__init__(val.size(-1))

		self.register_buffer('val', val)

	def update(self, val):
		self.val = val.to(self.val.device)

	def forward(self, t, q):
		return self.val



class SimpleLimiter(Controller):
	def __init__(self, force, index, range=1., offset=None, tolerance=0.01):

		try:
			len(index)
		except TypeError:
			index = [index]

		super().__init__(len(index))

		self.force = force # given (q[index] - offset)/(range + tol) compute force due to contact with limit at [-1,1]

		self.index = index

		self.range = range
		self.offset = offset

		self.tol = tolerance

	def forward(self, t, q):

		x = q[...,self.index]

		if self.offset is not None:
			x -= self.offset
		x /= (self.range + self.tol)

		u = self.force(x)
		return u

class PV_Limiter(Controller):
	def __init__(self, force, pos_index, vel_index, range=1., offset=None, tolerance=0.01):

		try:
			len(pos_index)
		except TypeError:
			pos_index = [pos_index]
			vel_index = [vel_index]

		super().__init__(len(pos_index))

		self.force = force # given (q[index] - offset)/(range + tol) compute force due to contact with limit at [-1,1]

		self.pos_index = pos_index
		self.vel_index = vel_index

		self.range = range
		self.offset = offset

		self.tol = tolerance

	def forward(self, t, q):

		x = q[...,self.pos_index]
		v = q[...,self.vel_index]

		if self.offset is not None:
			x -= self.offset
		x /= (self.range + self.tol)

		u = self.force(x, v)
		return u

# Forces, eg. for Limiter

class Force(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

class NonMechanical_Force(Force):
	def forward(self, x, v):
		raise NotImplementedError

class Mechanical_Force(Force):
	def forward(self, x):
		raise NotImplementedError

class PowerForce(Mechanical_Force):
	def __init__(self, power=2., coeff=-1., pre=None): # by default this is a restorative, harmonic force

		try:
			dim = len(power)
		except:
			dim = 1

		super().__init__(dim)

		self.power = power
		self.coeff = coeff
		self.pre = pre # eg. abs

	def forward(self, x):

		if self.pre is not None:
			x = self.pre(x)

		return self.coeff * x**self.power

class CutPowerForce(Mechanical_Force): # TODO: sort of a hack
	def __init__(self, power=1., coeff=-1., pre=None): # by default this is a restorative, harmonic force

		try:
			dim = len(power)
		except:
			dim = 1

		super().__init__(dim)

		self.power = power
		self.coeff = coeff
		self.pre = pre # eg. abs

	def forward(self, x):

		if self.pre is not None:
			x = self.pre(x)

		f = self.coeff * x**self.power
		f[x.abs()<1] = 0.
		return f