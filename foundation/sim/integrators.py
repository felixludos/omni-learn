import numpy as np
import torch

class Integrator(object):
	
	def __init__(self, dynamics, u_0, timestep):
		self.dynamics = dynamics
		self.u_0 = u_0
		self.timestep = timestep
		
		self.time = 0
		self.u = None
		
	def reset(self, u_0=None):
		if u_0 is not None:
			self.u_0 = u_0
		self.u = self.u_0
		
		assert self.u is not None
		
		self.time = 0
		
	def step(self, n_steps=1):
		raise NotImplementedError
	
class Second_Order_Integrator(Integrator):
	def __init__(self, dynamics, u_0, du_0, timestep):
		super(Second_Order_Integrator, self).__init__(dynamics, u_0, timestep)
		
		self.du_0 = du_0
		
		self.du = None
		
	def reset(self, u_0=None, du_0=None):
		super(Second_Order_Integrator, self).reset(u_0)
		if du_0 is not None:
			self.du_0 = du_0
		self.du = self.du_0
		
		assert self.du is not None
		
	def step(self, n_steps=1):
		raise NotImplementedError
	
	
class Velocity_Verlet(Second_Order_Integrator):
	
	def __init__(self, dynamics, u_0, du_0, timestep, coord_jacobian=None):
		super(Velocity_Verlet, self).__init__(dynamics, u_0, du_0, timestep)

		self.j = None
		if coord_jacobian is not None:
			self.j = coord_jacobian
		
	def reset(self, u_0=None, du_0=None):
		super(Velocity_Verlet, self).reset(u_0, du_0)
		
		self.ddu = self.dynamics(self.time, self.u, self.du)
		
	def step(self, n_steps=1):
		for _ in range(n_steps):
			# single verlet step
			half_vel = self.du + self.ddu * self.timestep / 2
			
			new_pos = self.u + half_vel * self.timestep if self.j is None else \
				self.u + half_vel * self.timestep * self.j(self.u) # for curvlinear coord systems
			
			new_acc = self.dynamics(self.time, new_pos, self.du)
			
			new_vel = half_vel + new_acc * self.timestep / 2
			
			self.u, self.du, self.ddu = new_pos, new_vel, new_acc
			self.time += self.timestep
			
		return self.u, self.du