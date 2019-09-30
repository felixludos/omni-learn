import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distrib
from foundation.foundation.models.probs import MLE, Joint_Distribution
from foundation.foundation import framework as fm
from foundation.foundation.models import networks as nets
import torch.multiprocessing as mp
from itertools import chain

# TODO: include exploration modules/schedules

class Model_Policy(fm.Policy):  # using function approximation as opposed to tabular policy
	def __init__(self, model, def_type='torch.FloatTensor', create_old=True):
		super(Model_Policy, self).__init__()
		self.model = model
		self.model.train()
		self.def_type = def_type
		if 'cuda' in def_type:
			self.model.cuda()
		
		self.old_model = None
		if create_old:
			self.old_model = copy.deepcopy(self.model)
			for old_p in self.old_model.parameters():
				old_p.requires_grad = False
	
	def __call__(self, obs):  # allows for batched obs (B x O)
		obs = obs.type(self.def_type)
		
		pi = self.get_pi(obs)
		
		if self.training_mode:
			action = pi.sample()
		else:
			action = MLE(pi)

		return action
	
	def train(self):
		super(Model_Policy, self).train()
		self.model.train()
	
	def eval(self):
		super(Model_Policy, self).eval()
		self.model.eval()
	
	def parameters(self):  # returns all of the policies parameters
		return self.model.parameters()
	
	def state_dict(self):
		return self.model.state_dict()
	
	def load_state_dict(self, state_dict):
		self.model.load_state_dict(state_dict)
		self.update_old_model()
	
	def get_pi(self, obs, include_old=False):  # returns full distribution (or policy if deterministic)
		raise Exception('not overridden')
	
	def update_old_model(self):
		self.old_model = copy.deepcopy(self.model)

class Normal_MultiCat_Policy(Model_Policy):  # more general standard policy (contains Gaussian, Categorical, and Multi_Cat_Policy)
	def __init__(self, model, num_normal, def_type='torch.FloatTensor', min_log_std=-3,
	             init_log_std=0, num_var=0):
		
		self.V = num_var
		self.I = num_normal
		if self.V > 0:
			self.C = (model.dout - self.I) // self.V

		if self.I > 0:
			model.log_std = torch.ones(self.I) * init_log_std
			if min_log_std is not None:
				model.log_std = nn.Parameter(model.log_std, requires_grad=True)
		
		super(Normal_MultiCat_Policy, self).__init__(model=model, def_type=def_type, create_old=True)
	
	def _split(self, logits):
		if self.V == 1:
			return distrib.Categorical(logits=logits)
		cats = [distrib.Categorical(logits=logit) for logit in logits.split(self.C, -1)]

		return Joint_Distribution(*cats)
	
	def _get_pi(self, model, obs, ):

		features = model(obs)

		normal = distrib.Normal(loc=features[:, :self.I],
		                                    scale=model.log_std.exp().unsqueeze(0)) if self.I > 0 else None

		#print('f',features[0, self.I:])
		mcat = self._split(features[:, self.I:]) if self.V > 0 else None
		#print('l',mcat.logits[0])

		if False:
			print(features)

			print(mcat)

			print(mcat.base[0].logits)

			quit()

			print('testing log std grads')
			from torch.autograd import grad

			# print(normal.mean, normal.variance)
			# quit()
			#
			#
			# t = torch.diag(model.log_std.exp())
			#
			# print(t)
			# print(t.requires_grad)
			# print(grad(t.sum(), model.log_std))
			#
			# quit()

			# print(grad(normal.loc.sum(), normal.loc))
			# print('var', normal.covariance_matrix)
			# print('type', type(normal.covariance_matrix))
			# print('req grad', normal.covariance_matrix.requires_grad)
			#
			# print('grad',grad(normal.covariance_matrix.sum(), model.log_std))
			#
			# #print(normal.loc, normal.variance)
			# print('quittin')
			# quit()

			# mu = torch.randn(2)
			# mu.requires_grad = True
			# sigma = torch.rand(2) * 10 + 1
			# sigma.requires_grad = True

			# n = distrib.MultivariateNormal(loc=mu, covariance_matrix=torch.diag(sigma))
			s = torch.ones(2)

			print(normal.log_prob(s))

			print(grad(normal.log_prob(s), [normal.loc, model.log_std]))

			# normal.log_prob(s).backward()
			# print('mu grad:',normal.loc.grad)
			# print('sigma grad:', normal.covariance_matrix.grad)
			quit()

			print(model.log_std)
			sample = torch.ones(2)
			# print(sample)
			print(normal.mean, normal.variance)
			lp = normal.log_prob(sample)
			print(torch.autograd.grad(lp, [normal.mean, normal.variance]))

			normal2 = distrib.Normal(loc=features[:, :self.I],
												 scale=(model.log_std * 2).exp()) if self.I > 0 else None

			# print(normal2.log_prob(sample))

			# print(model.log_std.grad)
			# print(normal.log_prob(normal.sample()).sum().backward())
			# print(model.log_std.grad)

			quit()


		if normal is not None and mcat is not None:
			return Joint_Distribution(normal, mcat)
		if normal is not None:
			return normal
		return mcat

	def get_pi(self, obs, include_old=False):

		obs = obs.view(-1, self.model.din)

		pi = self._get_pi(self.model, obs)

		if not include_old:
			return pi

		return pi, self._get_pi(self.old_model, obs)

	# def _split_features(self, features, log_std):
	# 	normal = distrib.MultivariateNormal(loc=features[:, :self.I],
	# 		                                covariance_matrix=torch.diag(log_std.exp())) if self.I > 0 else None
	# 	mcat = self._split(features[:, self.I:]) if self.V > 0 else None
	#
	# 	if normal is not None and mcat is not None:
	# 		return Joint_Distribution(normal, mcat)
	# 	if normal is not None:
	# 		return normal
	# 	return mcat
	#
	# def get_pi(self, obs, include_old=False):
	# 	obs = obs.view(-1, self.model.din)
	#
	# 	features = self.model(obs)
	#
	# 	pi = self._split_features(features, self.model.log_std)
	#
	# 	if not include_old:
	# 		return pi
	#
	# 	pi_old = self._split_features(features.detach(), self.model.log_std.detach())
	#
	# 	return pi, pi_old


class Gaussian_Policy(Normal_MultiCat_Policy):
	def __init__(self, model,
				 min_log_std=-3, # is None if log_std is constant
				 init_log_std=0,
				 def_type='torch.FloatTensor'):
		super(Gaussian_Policy,self).__init__(model, num_normal=model.dout, def_type=def_type, min_log_std=min_log_std, init_log_std=init_log_std)


class MultiCat_Policy(Normal_MultiCat_Policy):
	def __init__(self, model, num_vars, def_type='torch.FloatTensor'):
		super(MultiCat_Policy, self).__init__(model, num_normal=0, num_var=num_vars, def_type=def_type)


class Categorical_Policy(MultiCat_Policy):
	def __init__(self, model, def_type='torch.FloatTensor'):
		super(Categorical_Policy, self).__init__(model, def_type=def_type, num_vars=1)



class Joint_Policy(fm.Policy):
	def __init__(self, policies, parallel=True):
		super(Joint_Policy, self).__init__()
		
		self.policies = policies
		
		self.pool = None
		if parallel:
			self.pool = mp.Pool(len(policies))
	
	def __call__(self, obs):
		raise Exception('not overridden')
	
	def get_pi(self, obs, get_old=False):
		raise Exception('not overridden')
	
	def parameters(self):
		return chain(*[policy.parameters() for policy in self.policies])
	
	def state_dict(self):
		return [policy.state_dict() for policy in self.policies]
	
	def load_state_dict(self, state_dict):
		for policy, state in zip(self.policies, state_dict):
			policy.load_state_dict(state)

class Branched_Policy(Joint_Policy): # all sub policies take in the full obs
	def __init__(self, policies, parallel=True, separate=False): # TODO: add head model to extract common features from obs
		super(Branched_Policy, self).__init__(policies, parallel=parallel)
		self.separate = separate

	def __call__(self, obs):
		if self.pool is None:
			actions = [policy(obs) for policy in self.policies]
		else:
			actions = self.pool.map(lambda policy: policy(obs), self.policies)
		if self.separate:
			return actions
		return torch.cat(actions, -1)

	def get_pi(self, obs, include_old=False):
		if self.pool is None:
			pis = [policy.get_pi(obs, include_old) for policy in self.policies]
		else:
			pis = self.pool.map(lambda policy: policy.get_pi(obs, include_old), self.policies)

		if include_old:
			return [Joint_Distribution(*model) for model in zip(*pis)]
		return Joint_Distribution(*pis)


# old

class Full_Gaussian_Policy(fm.Policy):
	def __init__(self): # predicts both mu and sigma through model
		pass