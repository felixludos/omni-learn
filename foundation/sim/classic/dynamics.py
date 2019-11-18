
import torch
from torch import nn
from torch.nn import functional as F


class Dynamics(nn.Module):

	def __init__(self, dim, detach=True):
		super().__init__()

		self.dim = dim
		self.detach_io = detach

	def energy(self, q):
		raise NotImplementedError

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
	             length=.5, gravity=10.,
                 mass_cart=1, mass_pole=.4,
	             fric_cart=0., fric_pole=0.,
	             inertia_coeff=1.,
	             **options):
		super().__init__(dim=4, **options) # dims [x, theta, dx, dtheta]

		self.actuator = cart_force # should only return 1 dims - frc_x
		
		self.mass_cart = mass_cart
		self.mass_pole = mass_pole
		self.inertia_coeff = inertia_coeff # moment of inertia of pendulum = (this) * m_p * l**2

		self.length = length
		self.gravity = gravity  # magnitude (should be positive) (always downward)

		self.fric_cart = fric_cart # friction coeff (mu) between cart and ground
		self.fric_pole = fric_pole # friction coeff (mu) between pole and cart
		self.friction_full = (fric_pole + fric_cart) > 0
		if self.friction_full:
			self.register_buffer('Nc', torch.tensor(1.))
			
		self.old_dynamics = False

	def energy(self, q):
		return self.kinetic_energy(q) + self.potential_energy(q)
	# def total_energy(self, q):
	# 	x, theta = q.narrow(-1, 0, 1), q.narrow(-1, 1, 1)
	# 	dx, dtheta = q.narrow(-1, 2, 1), q.narrow(-1, 3, 1)
	#
	# 	m_p, m_c, l, g = self.mass_pole, self.mass_cart, self.length, self.gravity
	# 	I = self.inertia_coeff
	#
	# 	costheta, sintheta = torch.cos(theta), torch.sin(theta)
	#
	# 	T = (m_c * dx.pow(2) + I * m_p * ((dx + l * dtheta * costheta).pow(2) + (l * dtheta * sintheta).pow(2))) / 2
	# 	V = m_p * g * l * (costheta + 1)
	#
	# 	return T + V
	
	def kinetic_energy(self, q):
		theta = q.narrow(-1, 1, 1)
		dx, dtheta = q.narrow(-1, 2, 1), q.narrow(-1, 3, 1)
		
		m_p, m_c, l, g = self.mass_pole, self.mass_cart, self.length, self.gravity
		I = self.inertia_coeff
		
		costheta, sintheta = torch.cos(theta), torch.sin(theta)
		
		T = (m_c * dx.pow(2) + I * m_p * ((dx + l * dtheta * costheta).pow(2) + (l * dtheta * sintheta).pow(2))) / 2
		
		# TODO: check the effect of I != 1 on this formula
		
		return T
	
	def potential_energy(self, q):
		x, theta = q.narrow(-1, 0, 1), q.narrow(-1, 1, 1)
		dx, dtheta = q.narrow(-1, 2, 1), q.narrow(-1, 3, 1)
		
		theta = q.narrow(-1, 1, 1)
		m_p, l, g = self.mass_pole, self.length, self.gravity
		
		V = m_p * g * l * (torch.cos(theta) + 1)
		
		# V = self.lim_V(self, x, theta, dx, dtheta, V)
		
		return V

	def forward(self, t, q):

		if q.ndimension() == 1:
			q = q.unsqueeze(0)
		elif q.ndimension() > 2:
			raise Exception('too many: {}'.format(q.shape))

		x, theta = q.narrow(-1,0,1), q.narrow(-1,1,1)
		dx, dtheta = q.narrow(-1,2,1), q.narrow(-1,3,1)

		sintheta, costheta = torch.sin(theta), torch.cos(theta)

		m_p, m_c, l, g = self.mass_pole, self.mass_cart, self.length, self.gravity
		I = self.inertia_coeff

		M = m_p + m_c
		U = self.actuator(t, q).view(-1,1) if self.actuator is not None else torch.zeros_like(x)
			
		if self.friction_full and self.old_dynamics:
			assert False
			
		elif self.friction_full:
			
			f_c, f_p = self.fric_cart, self.fric_pole
			
			ddtheta = ((-U + f_c*dx  - m_p*l*dtheta**2*sintheta)*costheta + M*g*sintheta - (1 + m_c/m_p)*f_p/l*dtheta) / (
					l*(m_c + m_p*sintheta**2))
			ddx = (U - f_c*dx + f_p/l*dtheta*costheta + m_p*sintheta*(l*dtheta**2 - g*costheta)) / (
					m_c + m_p*sintheta**2)

		elif self.old_dynamics:
			assert False

		else:
			ddtheta = ((-U-m_p*l*dtheta**2*sintheta)*costheta + M*g*sintheta)/(
							l*(m_c + m_p*sintheta**2))
			ddx = (U + m_p*sintheta*(l*dtheta**2 - g*costheta))/ (
						m_c + m_p*sintheta**2)
			
		dq = torch.cat([dx, dtheta, ddx, ddtheta], -1)

		return dq.squeeze()

# old cartpole dynamics
# with friction:

# compute_ddtheta = lambda sgn: (g * sintheta + costheta * ((-U + lim_theta - m_p * l * dtheta ** 2 * (
# 			sintheta + f_c * sgn * costheta)) / M + f_c * g * sgn) - f_p * dtheta / (m_p * l)) / (
# 		                              l * (4. / 3 - (m_p * costheta / M) * (costheta - f_c * sgn)))
#
# f_c, f_p = self.fric_cart, self.fric_pole
# Nc = self.Nc
# sgn = (Nc * dx).sign()
# ddtheta = compute_ddtheta(sgn)
#
# Nc = M * g - m_p * l * (ddtheta * sintheta + dtheta ** 2 * costheta)
#
# if (Nc < 0).any():
# 	print('Bad: {}'.format(Nc))
#
# if ((self.Nc * Nc) < 0).any():  # Nc changes sign
# 	# recompute all
# 	sgn = (Nc * dx).sign()
# 	ddtheta = compute_ddtheta(sgn)
# 	Nc = M * g - m_p * l * (ddtheta * sintheta + dtheta ** 2 * costheta)
#
# self.Nc = Nc
#
# ddx = (U + lim_x + m_p * l * (dtheta ** 2 * sintheta - ddtheta * costheta) - f_c * Nc * sgn) / M

# without friction:

# ddtheta = (g * sintheta + costheta * (-U - m_p * l * dtheta ** 2 * sintheta) / M) / (
# 		l * (I - m_p * costheta ** 2 / M))
# ddx = (U + m_p * l * (dtheta ** 2 * sintheta - ddtheta * costheta)) / M






