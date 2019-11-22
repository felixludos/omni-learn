import numpy as np
import torch
import torch.nn.functional as F

# control algorithm
class Controller(object):

	def __init__(self, dynamics, cost, dt):
		self._dynamics = dynamics
		self._cost = cost
		self._dt = dt

	def reset(self, u):
		self._u_opt = u.clone()

	def __call__(self, x, t): # apply policy
		raise NotImplementedError

	def eval_trajectory_cost(self, x_traj, u_seq): # len(xtraj) + 1 == len(useq)
		J = sum([self._cost.eval(x,u) for x,u in zip(x_traj, u_seq)])

		# terminal state
		J += self._cost.eval(x_traj[-1], terminal=True)

		return J

	def eval_policy(self, x_traj, u_seq): # update policy based on trajectory (for iterative mpc)
		raise NotImplementedError

	def optimize(self, x, horizon): # learn policy for next 'horizon' steps
		raise NotImplementedError

class MPPI(Controller):

	def __init__(self, dynamics, cost, dt, stoc_proc=None, lmbda=1., n_rollouts=1000, std=1.):
		super(MPPI, self).__init__(dynamics=dynamics, cost=cost, dt=dt)

		self._n_rollouts = n_rollouts
		self._stoc_proc = stoc_proc
		self._std = std
		self._lambda = lmbda
		self._cost = cost
		self._u_opt = None

	def __call__(self, x, t):
		return self._u_opt[0]

	def optimize(self, state, horizon):

		C = self._u_opt.size(-1)

		U = self._u_opt.clone().unsqueeze(1) # T x 1 x C
		x0 = state.expand(self._n_rollouts, -1).contiguous() # K x S

		if self._stoc_proc is not None:
			noise = torch.from_numpy(self._stoc_proc.sequence(U.size(0))).float().cuda()  # T x K x C
		else:
			noise = torch.randn(U.size(0), self._n_rollouts, C).cuda() * self._std  # T x K x C

		X = self._dynamics.sequence(x0, U + noise, ret_all=True).detach()  # T x K x S

		cost = self._cost.eval_seq(X.permute(1,0,2))
		cost -= cost.min()
		w = F.softmin(cost / self._lambda, 0)

		self._u_opt += (w.view(1, -1, 1) * noise).sum(1)

		return True

class iLQR(Controller):

	def __init__(self, dynamics, cost, dt, max_reg=1e10):
		super(iLQR, self).__init__(dynamics=dynamics, cost=cost, dt=dt)

		self._reg_min = 1e-6
		self._reg_factor = 10
		self._reg_max = 1000

		# Regularization terms: Levenberg-Marquardt parameter.
		self._lm_reg = -1
		# See II F. Regularization Schedule.
		self._mu = 1.0
		self._mu_min = 1e-6
		self._mu_max = max_reg
		self._delta_0 = 2.0
		self._delta = self._delta_0
		self._alphas = 1.1**(-torch.arange(10).cuda()**2)
		self._alpha_idx = 0

		self._tol = 1e-6

		self._policy = None

	def _backward_pass(self, x_traj, u_seq):
		# traj has length time_horizon+1
		# u_seq has length time_horizon

		time_horizon = u_seq.size(0)
		k = [None] * time_horizon
		K = [None] * time_horizon
		reg = self._mu * torch.eye(x_traj.size(1)).cuda()

		Vxx = self._cost.dxx_terminal(x_traj[-1, :])
		Vx = self._cost.dx_terminal(x_traj[-1, :])

		for t in reversed(range(time_horizon)):
			xt, ut = x_traj[t, :], u_seq[t, :]
			dx_cost, du_cost = self._cost.dx(xt), self._cost.du(ut)
			dxx_cost, duu_cost = self._cost.dxx(xt), self._cost.duu(ut)
			dux_cost = self._cost.dux(xt, ut)

			dx_dynT, du_dynT = self._dynamics.jacobians(xt, ut) # returns transposed jacobians
			dx_dyn, du_dyn = dx_dynT.transpose(0,1), du_dynT.transpose(0, 1)

			Qx = dx_cost + dx_dynT @ Vx
			Qu = du_cost + du_dynT @ Vx
			Qxx = dxx_cost + dx_dynT @ Vxx @ dx_dyn
			Qux = dux_cost + du_dynT @ (Vxx + reg) @ dx_dyn
			Quu = duu_cost + du_dynT @ (Vxx + reg) @ du_dyn

			if self._lm_reg > 0:  # use Levenberg-Marquadt trick
				U, S, V = torch.svd(Quu)
				S[S < 0] = 0
				iQuu = U @ (1. / torch.diag(S + self._lm_reg)) @ V.transpose(0, 1)
			else:
				iQuu = torch.inverse(Quu)

			k[t] = -iQuu @ Qu
			K[t] = -iQuu @ Qux

			Vx = Qx + K[t].transpose(0,1) @ Quu @ k[t]
			Vx += K[t].transpose(0,1) @ Qu + Qux.transpose(0,1) @ k[t]

			Vxx = Qxx + K[t].transpose(0,1) @ Quu @ K[t]
			Vxx += K[t].transpose(0,1) @ Qux + Qux.transpose(0,1) @ K[t]
			Vxx = 0.5 * (Vxx + Vxx.transpose(0,1))

		return torch.stack(k), torch.stack(K)

	def _simulate_feedback_policy(self, x_des, u_seq, k, K, alpha):
		x_new = torch.zeros_like(x_des)
		u_new = torch.zeros_like(u_seq)
		time_horizon = u_seq.size(0)

		x_new[0] = x_des[0]
		for t in range(time_horizon):
			u_new[t] = u_seq[t] + alpha * (k[t] + K[t].dot(x_new[t] - x_des[t]))
			x_new[t+1] = self._dynamics(x_new[t], u_new[t])

		return x_new, u_new

	def __call__(self, x, t):
		return self._policy(x,t)

	def eval_policy(self, x_traj, u_seq):
		J_new = self.eval_trajectory_cost(x_traj, u_seq)
		converged = False
		accept = False

		if J_new < self._J_opt:
			if np.abs((self._J_opt - J_new) / self._J_opt) < self._tol:
				converged = True
			else:
				# decrease regularization
				self._delta = min(1.0, self._delta) / self._delta_0
				self._mu *= self._delta
				if self._mu <= self._mu_min:
					self._mu = 0.0
				accept = True

			self.J_opt = J_new
			self._u_opt = u_seq

		self._alpha_idx += 1 # decrease alpha
		if self._alpha_idx == len(self._alphas) and not accept: # already at lowest alpha
			self._delta = max(1.0, self._delta) * self._delta_0
			self._mu = max(self._mu_min, self._mu * self._delta)
			if self._mu_max and self._mu >= self._mu_max:
				print("regularization term too large, quitting")
		self._alpha_idx = min(self._alpha_idx, len(self._alphas) - 1)

		return J_new

	def optimize(self, x0, time_horizon):

		# get previous optimal command sequence
		u_seq = self._u_opt.clone()

		x_traj = self._dynamics.sequence(x0, u_seq).squeeze()
		self._J_opt = self.eval_trajectory_cost(x_traj, u_seq)

		accept = False
		k, K = self._backward_pass(x_traj, u_seq)
		alpha = self._alphas[self._alpha_idx]

		self._policy = lambda x, t: u_seq[t] + alpha * (k[t] + K[t] @ (x - x_traj[t]))

