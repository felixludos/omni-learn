
import torch
from torch import nn

class PID_Controller(nn.Module): # discrete time
	def __init__(self, kp=1, ki=0, kd=0, timestep=None):
		super().__init__()
		self.kp, self.ki, self.kd = kp, ki, kd

		if timestep is not None:
			self.ki = self.kp / timestep
			self.kd = self.kp * timestep

	#         self.register_buffer('prev', torch.zeros(dim))
	#         self.register_buffer('sum', torch.zeros(dim))

	def reset(self):
		if hasattr(self, 'prev'):
			del self.prev
		if hasattr(self, 'sum'):
			del self.sum

	def forward(self, error):
		ctrl = self.kp * error

		if self.ki != 0:
			if not hasattr(self, 'sum'):
				self.register_buffer('sum', torch.zeros_like(error))
			ctrl += self.ki * self.sum

			self.sum += error

		if self.kd != 0:
			if hasattr(self, 'prev'):
				ctrl += self.kd * (error - self.prev)
				self.prev.copy_(error)
			else:
				self.register_buffer('prev', torch.zeros_like(error))

		return ctrl

class Soft_3D_Constraint(PID_Controller):
	def __init__(self, ref, direction,
				 left=True, right=True, **params):
		super().__init__(**params)

		self.register_buffer('ref', ref)

		direction /= direction.norm(p=2)

		self.register_buffer('direction', direction)

		# act when the x is _ of ref
		self.left = left
		self.right = right

		assert left or right

	def update_ref(self, ref):
		self.ref[:] = ref

	def forward(self, x):

		err = self.direction @ (self.ref - x)

		if (self.left and self.right) or (self.left and err > 0) or (self.right and err < 0):
			u = super().forward(err)
		else:
			#             print('no action')
			u = 0
			self.reset()

		return u * self.direction