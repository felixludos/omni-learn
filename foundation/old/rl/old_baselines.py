import numpy as np
import torch
import copy
from foundation.foundation import framework as fm
from foundation.foundation.models import networks as nets
from torch.utils.data import DataLoader, TensorDataset
from foundation.foundation import util

def feature_extractor(X, lim=None, ones=True, obs_order=1, time_order=1):
	
	T, O = X.size() # T x O

	if lim is not None:
		X = X.clamp(-lim, lim)

	ts = torch.arange(0,T).float().view(-1,1) / 1000 # timestep
	ts = ts.type_as(X)
	
	terms = [X**(n+1) for n in range(obs_order)]
	terms.extend([ts**(n+1) for n in range(time_order)])
	
	if ones:
		terms.append(torch.ones_like(ts))
	
	return torch.cat(terms, 1)

class MLP_Baseline(fm.Baseline):
	def __init__(self, obs_dim, time_order=1, scale=None, hidden_dims=[], momentum=0.5, nesterov=True, epochs=5, batch_size=64,
	             optim_type='rmsprop', lr=1e-3, weight_decay=1e-5, loss_type='mse', nonlin='elu'):
		super(MLP_Baseline, self).__init__()
		
		self.model = nets.MLP(input_dim=obs_dim+time_order, output_dim=1, hidden_dims=hidden_dims, nonlinearity=nonlin)

		self.batch_size = batch_size
		self.epochs = epochs
		self.scale = scale
		
		self.criterion = nets.get_loss_type(loss_type)
		self.optim = nets.get_optimizer(optim_type, self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
		
		#print(self.model, self.epochs, self.scale, self.criterion, self.scale, self.optim)
		#quit()
	
	def _features(self, obs): # T x O
		return feature_extractor(obs, ones=False, time_order=1)
	
	def train_step(self, paths):
		X_data = torch.stack([self._features(obs) for obs in paths.observations]).view(-1, self.model.din).detach()
		if self.scale is None:
			self.scale = torch.stack(paths.returns).view(-1, 1).detach().mean().abs() # set scale once and then never again
		Y_data = torch.stack(paths.returns).view(-1, 1).detach() / self.scale
		
		stats = util.StatsMeter('error-before', 'error-final')
		stats.update('error-before', self.criterion(self.model(X_data), Y_data).item())
		
		data = DataLoader(TensorDataset(X_data, Y_data), batch_size=self.batch_size, shuffle=True)
		
		for epoch in range(self.epochs):  # epochs
			
			for X, Y in data:  # iters per epoch
				
				loss = self.criterion(self.model(X), Y)
				
				#print(loss.item())
				
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()
		
		stats.update('error-final', self.criterion(self.model(X_data), Y_data).item())
		return stats
	
	def __call__(self, obs, act=None):  # path T x O
		X = self._features(obs).view(-1, self.model.din)
		return self.model(X).view(obs.size(0)) * (1 if self.scale is None else self.scale)

class LinearBaselineLeastSqrs(fm.Baseline):
	def __init__(self, obs_dim, time_order=1, obs_order=1, reg_coeff=1e-8, def_type='torch.FloatTensor'):
		super(LinearBaselineLeastSqrs, self).__init__()
		self._t_order = time_order
		self._o_order = obs_order
		self._reg_coeff = reg_coeff
		self._coeffs = torch.zeros(obs_dim*obs_order + time_order + 1,1).type(def_type)

		self.criterion = nets.get_loss_type('mse')
	
	def __call__(self, obs):  # obs_batches B x T x O
		return self._features(obs).mm(self._coeffs).view(-1)

	def _features(self, obs):
		return feature_extractor(obs, time_order=self._t_order, obs_order=self._o_order)

	def train_step(self, observations, returns):

		X = torch.cat([self._features(obs) for obs in observations])
		Y = torch.cat(returns).view(-1, 1)

		A = X.transpose(0,1).mm(X).cpu().detach().numpy()
		b = X.transpose(0,1).mm(Y).cpu().detach().numpy()
		rI = np.eye(X.size(-1)) * self._reg_coeff

		stats = util.StatsMeter('error-before', 'error-final')
		stats.update('error-before', ((X @ self._coeffs - Y)**2).mean().item())

		reg_coeff = self._reg_coeff
		for _ in range(10):
			soln = np.linalg.lstsq(
				A + rI,
				b
			)[0]
			if np.isfinite(soln).all():
				break
			rI *= 10

		# set new coeffs
		self._coeffs = torch.from_numpy(soln).type_as(X)

		stats.update('error-final', ((X @ self._coeffs - Y) ** 2).mean().item())
		return stats
