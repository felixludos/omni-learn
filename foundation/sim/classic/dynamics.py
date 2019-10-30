
import torch
from torch import nn
from torch.nn import functional as F


class Dynamics(nn.Module):

	def __init__(self, dim, detach=True):
		super().__init__()

		self.dim = dim
		self.detach_io = detach

	def __call__(self, t, q):
		if self.detach_io:
			t = t.detach()
			q = q.detach()
		dq = super().__call__(t, q)
		if self.detach_io:
			dq = dq.detach()
		return dq

	def forward(self, t, q):
		return q

class Controlled_Dynamics(Dynamics):
	def __init__(self, dim, controller, **options):
		super().__init__(dim, **options)

		self.controller = controller

class Cartpole_Dynamics(Dynamics):
	'''
	Using the equations derived by Razvan V. Florian in "Correct equations for the dynamics of the cart-pole system"
	see https://coneural.org/florian/papers/05_cart_pole.pdf
	'''

	def __init__(self, cart_force=None,
	             mass_cart=1., mass_pole=1.,
	             length=1., gravity=1.,
	             fric_cart=0., fric_pole=0.,
	             limit=None,
	             **options):
		super().__init__(dim=4, **options) # dims [x, theta, dx, dtheta]

		self.controller = cart_force # should only return 1 dims - frc_x

		self.mass_cart = mass_cart
		self.mass_pole = mass_pole

		self.length = length
		self.gravity = gravity  # magnitude (should be positive) (always downward)

		self.limit = limit

		self.fric_cart = fric_cart # friction coeff (mu) between cart and ground
		self.fric_pole = fric_pole # friction coeff (mu) between pole and cart
		self.friction_full = (fric_pole + fric_cart) > 0
		if self.friction_full:
			self.register_buffer('Nc', torch.tensor(1.))

	def forward(self, t, q):

		if q.ndimension() == 1:
			q = q.unsqueeze(0)
		elif q.ndimension() > 2:
			raise Exception('too many: {}'.format(q.shape))

		x, theta = q.narrow(-1,0,1), q.narrow(-1,1,1)
		dx, dtheta = q.narrow(-1,2,1), q.narrow(-1,3,1)

		sintheta, costheta = torch.sin(theta), torch.cos(theta)

		m_p, m_c, l, g = self.mass_pole, self.mass_cart, self.length, self.gravity

		M = m_p + m_c
		U = self.controller(t, q).view(-1,1) if self.controller is not None else torch.zeros_like(x)

		if self.friction_full:
			compute_ddtheta = lambda sgn: (g*sintheta + costheta*((-U - m_p*l*dtheta**2*(sintheta + f_c*sgn*costheta))/M + f_c*g*sgn) - f_p*dtheta/(m_p*l)) / (
													l*(4./3 - (m_p*costheta/M)*(costheta - f_c*sgn)))

			f_c, f_p = self.fric_cart, self.fric_pole
			Nc = self.Nc
			sgn = (Nc * dx).sign()
			ddtheta = compute_ddtheta(sgn)

			Nc = M*g - m_p*l*(ddtheta*sintheta + dtheta**2*costheta)

			if (Nc < 0).any():
				print('Bad: {}'.format(Nc))

			if ((self.Nc*Nc) < 0).any(): # Nc changes sign
				# recompute all
				sgn = (Nc * dx).sign()
				ddtheta = compute_ddtheta(sgn)
				Nc = M*g - m_p*l*(ddtheta*sintheta + dtheta**2*costheta)

			self.Nc = Nc

			ddx = (U + m_p*l*(dtheta**2*sintheta - ddtheta*costheta) - f_c*Nc*sgn)/M

		else:
			ddtheta = (g*sintheta + costheta*(-U-m_p*l*dtheta**2*sintheta)/M) / (
									l*(4./3 - m_p*costheta**2/M))
			ddx = (U + m_p*l*(dtheta**2*sintheta - ddtheta*costheta)) / M

		if self.limit is not None: # This is quite a hack, but it looks reasonable...
			sel = (x.abs()>self.limit)*((x*dx).sign()>0)
			dx[sel] /= 10

		dq = torch.cat([dx, dtheta, ddx, ddtheta], -1)

		return dq.squeeze()







