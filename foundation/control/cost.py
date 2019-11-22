import numpy as np
import torch

# cost
class QuadraticCost(object):

	def __init__(self, Q, R, x_target, state_dim, action_dim, Q_terminal=None):
		'''

		:param Q:
		:param R:
		:param x_target: x_dim vector
		:param Q_terminal:
		'''

		self.Q = Q
		self.R = R
		self.Q_terminal = Q.clone() if Q_terminal is None else Q_terminal

		self.S = state_dim
		self.C = action_dim

		self.x_target = x_target

	def eval(self, x, u=None, terminal=False):
		x_err = x - self.x_target

		if terminal: # terminal cost doesn't include control
			return 0.5 * x_err @ self.Q_terminal @ x_err
		return 0.5 * (x_err @ self.Q @ x_err + u @ self.R @ u)

	def eval_seq(self, x, u=None):  # B x T+1 x S, B x T x C
		x = x[:, :-1]
		xt = x[:, -1]
		B = x.size(0)

		x = x.contiguous().view(-1, 1, self.S)

		q = x.contiguous().view(-1, 1, self.S) @ self.Q @ x.view(-1, self.S, 1)

		if u is not None:
			u = u.view(-1, 1, self.C)
			q += u.view(-1, 1, self.C) @ self.R @ u.view(-1, self.C, 1)

		phi = xt.view(-1, 1, self.S) @ self.Q_terminal @ xt.view(-1, self.S, 1)

		return (q.view(B, -1).sum(-1) + phi.view(B)) / 2.

	def du(self, u):
		return self.R @ u

	def dx(self, x):
		x_err = x - self.x_target
		return self.Q @ x_err

	def dx_terminal(self, x):
		x_err = x - self.x_target
		return self.Q_terminal @ x_err

	def dxu(self, x, u):
		return self.dux(x,u)

	def dux(self, x, u):
		return torch.zeros_like(x)

	def duu(self, u):
		return self.R

	def dxx(self, x):
		return self.Q

	def dxx_terminal(self, x):
		return self.Q_terminal