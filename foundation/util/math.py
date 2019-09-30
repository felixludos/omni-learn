
import sys, os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as distrib
from torch.distributions.utils import lazy_property

#####################
# Simple math
#####################

def atanh(x): # probably shouldnt be used
	return (1+x).div(1-x).log() / 2

def cum_sum(xs, cut_prev=False):
	out = [xs[0]]

	for x in xs[1:]:
		if cut_prev:
			out.append(x + out[-1].detach())
		else:
			out.append(x + out[-1])

	return torch.stack(out)

def cum_prod(xs, cut_prev=False):
	out = [xs[0]]

	for x in xs[1:]:
		if cut_prev:
			out.append(x * out[-1].detach())
		else:
			out.append(x * out[-1])

	return torch.stack(out)

#####################
# Neural Networks
#####################

def get_loss_type(name, **kwargs):

	if not isinstance(name, str):
		return name

	if name == 'mse':
		return nn.MSELoss(**kwargs)
	elif name == 'l1':
		return nn.L1Loss(**kwargs)
	elif name == 'huber':
		return nn.SmoothL1Loss(**kwargs)
	elif name == 'nll':
		print('WARNING: should probably use cross-entropy')
		return nn.NLLLoss(**kwargs)
	elif name == 'cross-entropy':
		return nn.CrossEntropyLoss(**kwargs)
	elif name == 'kl-div':
		return nn.KLDivLoss(**kwargs)
	elif name == 'bce':
		#print('WARNING: should probably use bce-log')
		return nn.BCELoss(**kwargs)
	elif name == 'bce-log':
		return nn.BCEWithLogitsLoss(**kwargs)
	else:
		assert False, "Unknown loss type: " + name

# Choose non-linearities
def get_nonlinearity(nonlinearity, dim=1, inplace=True):

	if nonlinearity is None:
		return None
	if not isinstance(nonlinearity, str):
		return nonlinearity

	if nonlinearity == 'prelu':
		return nn.PReLU()
	elif nonlinearity == 'lrelu':
		return nn.LeakyReLU()
	elif nonlinearity == 'relu':
		return nn.ReLU(inplace=inplace)
	elif nonlinearity == 'tanh':
		return nn.Tanh()
	elif nonlinearity == 'log-softmax':
		return nn.LogSoftmax(dim=dim)
	elif nonlinearity == 'softmax':
		return nn.Softmax(dim=dim)
	elif nonlinearity == 'softmax2d':
		return nn.Softmax2d()
	elif nonlinearity == 'softplus':
		return nn.Softplus()
	elif nonlinearity == 'sigmoid':
		return nn.Sigmoid()
	elif nonlinearity == 'elu':
		return nn.ELU(inplace=inplace)
	else:
		assert False, "Unknown nonlin type: " + nonlinearity

def get_normalization(norm, num, **kwargs):
	if norm == 'batch':
		return nn.BatchNorm2d(num, **kwargs)
	if norm == 'instance':
		return nn.InstanceNorm2d(num, **kwargs)



#####################
# Randomness and Noise
#####################

class OUNoise(nn.Module):
	"""docstring for OUNoise"""
	def __init__(self, batch_size, dim, mu=0, theta=1, sigma=1, batch_first=True):
		super(OUNoise, self).__init__()
		self.dim = dim
		self.batch_size = batch_size
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.stack_dim = 1 if batch_first else 0
		self.reset()

	def reset(self):
		state = torch.ones(self.batch_size, self.dim) * self.mu
		self.register_buffer('state', state)

	def forward(self, seq=1, reset=False):
		if reset:
			self.reset()

		states = [self.state]
		for k in range(seq):
			dx = self.theta * (self.mu - states[-1]) + self.sigma * torch.randn(self.batch_size, self.dim).type_as(self.state)
			states.append(states[-1] + dx)

		self.state = states[-1]
		states = torch.stack(states, self.stack_dim) # batch x seq x dim or seq x batch x dim
		return states.squeeze(1).squeeze(0)

def set_seed(seed=None):
	if seed is None:
		seed = get_random_seed()
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	return seed

def get_random_seed():
	return np.frombuffer(np.random.bytes(4), dtype=np.int32)[0]


#####################
# Linear Systems
#####################

def mat_sum(mats):
	out = mats[0]
	for mat in mats[1:]:
		out = mat @ out
	return out

def cum_matmul_(mats):
	for i in range(1,len(mats)):
		mats[i] = mats[i] @ mats[i-1]
	return mats

def cum_matmul(mats, cut_prev=False):
	out = [mats[0]]
	for m in mats[1:]:
		if cut_prev:
			out.append(m @ out[-1].detach())
		else:
			out.append(m @ out[-1])
	return torch.stack(out)

def solve(A, b, out=None, bias=True, reg=0):
	'''
	Solves for x to minimize (Ax-b)^2
	for some matrix A and vector b
	x is returned as a linear layer (either with or without a bias term)
	Will update out if given, otherwise the output will be a new linear layer
	:param A: D x N pytorch tensor
	:param b: N x K pytorch tensor
	:param out: instance of torch.nn.Linear(D,K)
	:param bias: learn a bias term in addition to weights
	:return: torch.nn.Linear(D, K) instance where the weights (and bias) solve Ax=b
	'''
	# A: M x N
	# b: N x K
	# x: M x K

	if reg > 0:
		R = A.t() @ A
		b = A.t() @ b
		A = R + reg*torch.eye(A.size(1)).type_as(A)

	# print(A.size(), b.size())

	if bias:
		A = torch.cat([A, torch.ones(*(A.size()[:-1] + (1,))).type_as(A)], -1)

	x, _ = torch.gels(b, A)

	if out is None:
		out = nn.Linear(A.size(-1) - 1, b.size(-1), bias=bias).to(A.device)

	out.weight.data.copy_(x[:A.size(-1) - 1].t())  # TODO: make sure this works both with and without bias

	if bias:
		out.bias.data.copy_(x[A.size(-1) - 1:A.size(-1), :].squeeze())

	return out

# def solve_reg(A, b, out=None, bias=True, l2_reg=0):
#
# 	R = A.t() @ A


#####################
# Affine transformations/conversions
#####################

def aff_transform(points, transforms):

	D = transforms.shape[-1] - 1

	R, t = transforms.narrow(-1, 0, D), transforms.narrow(-1, D, 1)
	# R = R.transpose(-1,-2)

	return R @ points + t


def aff_compose(*poses):
	if len(poses) == 1:
		return poses[0]
	D = poses[0].shape[-1] - 1
	oR, ot = poses[0].narrow(-1, 0, D), poses[0].narrow(-1, D, 1)
	for Rt in poses[1:]:
		R, t = Rt.narrow(-1, 0, D), Rt.narrow(-1, D, 1)
		ot = oR @ t + ot
		oR = oR @ R
	return torch.cat([oR, ot], -1)


def aff_invert(pose): # just for rotation + translations
	D = pose.shape[-1] - 1
	R, t = pose.narrow(-1, 0, D), pose.narrow(-1, D, 1)
	Rp = R.transpose(-1, -2)
	return torch.cat([Rp, -Rp @ t], -1)

def aff_negate(pose): # just for rotation + translations
	D = pose.shape[-1] - 1
	R, t = pose.narrow(-1, 0, D), pose.narrow(-1, D, 1)
	Rp = R.transpose(-1, -2)
	return torch.cat([Rp, -t], -1)

def aff_add(*poses):
	if len(poses) == 1:
		return poses[0]
	D = poses[0].shape[-1] - 1
	oR, ot = poses[0].narrow(-1, 0, D), poses[0].narrow(-1, D, 1)
	for Rt in poses[1:]:
		R, t = Rt.narrow(-1, 0, D), Rt.narrow(-1, D, 1)
		oR = oR @ R
		ot = t + ot  ## <---------- no rotation/translation coupling!
	return torch.cat([oR, ot], -1)


def aff_cum_sum(poses):
	for i in range(1,len(poses)):
		poses[i] = aff_add(poses[i],poses[i-1])
	return poses

def aff_cum_compose(poses): # WARNING: be careful about ordering
	for i in range(1,len(poses)):
		poses[i] = aff_compose(poses[i],poses[i-1])
	return poses


def se3_euler2Rt(euler): # x,y,z,wx,wy,wz (psi,theta,phi)
	trans, rot = euler.narrow(-1, 0, 3), euler.narrow(-1, 3, 3)
	rot = euler2mat(rot)

	return torch.cat([rot, trans.unsqueeze(-1)], -1)


def se3_Rt2euler(Rt): # doesn't always work with grads
	Rt = Rt.unsqueeze(0)

	angles = mat2euler(Rt.narrow(-1, 0, 3))
	trans = Rt.narrow(-1, 3, 1)

	euler = torch.cat([trans.squeeze(-1), angles], -1)

	return euler.squeeze(0)

def se3_quat2Rt(se3quats): # qw, qx, qy, qz, x, y, z

	quats = se3quats.narrow(-1, 0, 4)
	trans = se3quats.narrow(-1, 4, 3).unsqueeze(-1)

	rots = quat2mat(quats)

	return torch.cat([rots,trans],-1)

#####################
# Rotations
#####################

def geodesic(R, eps=1e-8):
	diag = torch.diagonal(R, 0, -1, -2).sum(-1)
	angles = torch.acos((diag - 1).div(2).clamp(-1 + eps, 1 - eps))
	return angles

def euler2mat(euler):
	psi, theta, phi = euler.narrow(-1, 0, 1), euler.narrow(-1, 1, 1), euler.narrow(-1, 2, 1)

	cp, ct, cf = torch.cos(psi), torch.cos(theta), torch.cos(phi)
	sp, st, sf = torch.sin(psi), torch.sin(theta), torch.sin(phi)

	rot = torch.cat([
		ct * cf, sp * st * cf - cp * sf, cp * st * cf + sp * sf,
		ct * sf, sp * st * sf + cp * cf, cp * st * sf - sp * cf,
		-st, sp * ct, cp * ct
	], -1)

	end_shape = rot.shape[:-1]

	return rot.view(end_shape + (3, 3))

def mat2euler(mat):
	rot = mat.narrow(-1, 0, 3)

	theta = -torch.asin(rot[..., 2, 0])

	ct = torch.cos(theta)
	psi = torch.atan2(rot[..., 2, 1] / ct, rot[..., 2, 2] / ct)
	phi = torch.atan2(rot[..., 1, 0] / ct, rot[..., 0, 0] / ct)

	sel = torch.isclose(rot[..., 2, 0], torch.tensor(1.).type_as(rot))
	if sel.any():
		theta[sel] = -np.pi / 2
		phi[sel] = 0
		psi[sel] = torch.atan2(-rot[sel, 1, 0], -rot[sel, 0, 2])

	sel = torch.isclose(rot[..., 2, 0], torch.tensor(-1.).type_as(rot))
	if sel.any():
		theta[sel] = np.pi / 2
		phi[sel] = 0
		psi[sel] = torch.atan2(rot[sel, 1, 0], rot[sel, 0, 2])

	return torch.stack([psi, theta, phi], -1)


def aa2mat(axisangle, eps=1e-8):

	angle = axisangle.norm(p=2, dim=-1, keepdim=True) + eps

	axis = axisangle / angle

	s = torch.sin(angle)
	c = torch.cos(angle)

	R = c.expand_as(axis).diag_embed(0)

	R += axis.unsqueeze(-1) * axis.unsqueeze(-2) * (1 - c.unsqueeze(-1))

	d = s * axis

	x, y, z = d.narrow(-1,0,1), d.narrow(-1,1,1), d.narrow(-1,2,1)
	o = torch.zeros(x.shape, device=x.device).float()

	R += torch.stack([
		torch.cat([o, -z, y],-1),
		torch.cat([z, o, -x], -1),
		torch.cat([-y, x, o], -1),
	], -2)

	return R

def mat2aa(R):
	raise NotImplementedError # requires batched eig - not available yet

def mat2c6d(R): # called g_gs in the paper https://arxiv.org/pdf/1812.07035.pdf
	return R[...,:-1]

def mat2quat(R):

	w = (1 + R[...,0,0] + R[...,1,1] + R[...,2,2]).sqrt().div(2)
	n = 4*w
	x = (R[..., 2, 1] - R[..., 1, 2]).div(n)
	y = (R[..., 0, 2] - R[..., 2, 0]).div(n)
	z = (R[..., 1, 0] - R[..., 0, 1]).div(n)

	return torch.stack([w,x,y,z],-1)

def c6d2mat(c6d): # called f_gs in the paper https://arxiv.org/pdf/1812.07035.pdf

	if c6d.shape[-1] == 6:
		shape = c6d.shape[:-1] + (3,2)
		c6d = c6d.view(*shape)

	assert c6d.shape[-2:] == (3,2), 'Only setup for converting 3d rotations: {}'.format(c6d.shape)

	a1, a2 = c6d.narrow(-1,0,1), c6d.narrow(-1,1,1)

	b1 = F.normalize(a1, dim=-2)
	b2 = F.normalize(a2 - ((b1 * a2).sum(-2,keepdim=True) * b1), dim=-2)

	b3 = b1.cross(b2, dim=-2)

	return torch.cat([b1,b2,b3],-1)



def quat2mat(quat): # w, x, y, z
	assert quat.size(-1) == 4,'wrong shape: {}'.format(quat.shape)

	quat = F.normalize(quat,p=2,dim=-1)

	w, x, y, z = quat.narrow(-1,0,1), quat.narrow(-1,1,1), quat.narrow(-1,2,1), quat.narrow(-1,3,1)

	x2, y2, z2 = x.pow(2), y.pow(2), z.pow(2)

	R1 = torch.cat([1-2*(y2+z2), 2*(x*y-z*w), 2*(x*z+y*w)],-1)
	R2 = torch.cat([2*(x*y+z*w), 1-2*(x2+z2), 2*(y*z-x*w)], -1)
	R3 = torch.cat([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x2+y2)], -1)

	return torch.stack([R1,R2,R3], -2)

def gnom2mat(gnom): # no 180 degree rotations
	assert gnom.size(-1) == 3, 'wrong shape: {}'.format(gnom.shape)

	w = torch.ones(gnom.shape[:-1]).type_as(gnom).unsqueeze(-1)
	return quat2mat(torch.cat([w,gnom],-1))

#####################
# Probabilities/Statistics
#####################

class Joint_Distribution(distrib.Distribution):
	def __init__(self, *base_distributions):
		super(Joint_Distribution, self).__init__()
		self.base = base_distributions

		self._check_base()

	def _check_base(self):
		self.douts = []
		self._batch_shape = self.base[0].batch_shape

		self._dtype = self.base[0].sample().type()

		for distr in self.base:
			assert type(distr) in _known_distribs, str(distr) + ' unknown'
			assert self.batch_shape == distr.batch_shape, str(self.batch_shape) + ' is not ' + distr.batch_shape

			new_type = distr.sample().type()
			if new_type < self._dtype:
				self._dtype = new_type

			try:
				event_shape = distr.event_shape[0]
			except IndexError:
				event_shape = 1

			self.douts.append(event_shape)

		self.E = sum(self.douts)
		self._event_shape = torch.Size([self.E]) if self.E > 1 else torch.Size([])

	@lazy_property
	def mean(self):
		means = []
		for p, s in zip(self.base, self.douts):
			try:
				if isinstance(p.mean, list):
					means.append(p.mean)
				elif p.mean.contiguous().view(-1)[0].item() == p.mean.contiguous().view(-1)[0]:
					means.append(p.mean)
				else:
					means.append(None)
			except AttributeError:
				means.append(None)
		return means

	@lazy_property
	def logits(self):
		logits = []
		for p, s in zip(self.base, self.douts):
			try:
				if isinstance(p.logits, list):  # p is a joint distribution?
					logits.append(p.logits)
				elif p.logits.contiguous().view(-1)[0].item() == p.logits.contiguous().view(-1)[0]:
					logits.append(p.logits)
				else:
					logits.append(None)
			except AttributeError:
				logits.append(None)
		return logits

	@lazy_property
	def probs(self):
		probs = []
		for p, s in zip(self.base, self.douts):
			try:
				if isinstance(p.logits, list):
					probs.append(p.probs)
				elif p.probs.contiguous().view(-1)[0].item() == p.probs.contiguous().view(-1)[0]:
					probs.append(p.probs)
				else:
					probs.append(None)
			except AttributeError:
				probs.append(None)
		return probs

	def log_prob(self, value, separate=False):
		vals = value
		if not isinstance(value, list):
			idx = 0
			vals = []
			for s in self.douts:
				vals.append(value[..., idx:idx + s] if s > 1 else value[..., idx])
				idx += s
		probs = [p.log_prob(v) for p, v in zip(self.base, vals)]
		return probs if separate else sum(probs)

	def sample(self, n=torch.Size([]), separate=False):
		if separate:
			return [p.sample(n) for p in zip(self.base)]
		samples = torch.cat(
			[p.sample(n).type(self._dtype) if s > 1 else p.sample(n).type(self._dtype).unsqueeze(-1) for p, s in
			 zip(self.base, self.douts)], -1)
		return samples.view(self._extended_shape(n))

	@lazy_property
	def mle(self, separate=False):
		if separate:
			return [MLE(p) for p, s in zip(self.base, self.douts)]
		mle = torch.cat([MLE(p).type(self._dtype) if s > 1 else MLE(p).unsqueeze(-1).type(self._dtype) for p, s in
						 zip(self.base, self.douts)], -1)
		return mle.view(self._extended_shape())

	def __repr__(self):
		return 'Joint_Distribution({})'.format(', '.join(map(str, self.base)))


_known_distribs = [distrib.MultivariateNormal, distrib.Categorical, Joint_Distribution]

def group_kl(p, q):
	return sum([distrib.kl.kl_divergence(p_i, q_i) for p_i, q_i in zip(p.base, q.base)])
distrib.kl.register_kl(Joint_Distribution, Joint_Distribution)(group_kl)

def standard_kl(p, q=None):
	mu, sigma = p.loc, p.scale
	return (mu.pow(2) - sigma.log() + sigma - 1) / 2
distrib.kl.register_kl(distrib.Normal, type(None))(standard_kl)


def MLE(q):
	if isinstance(q, distrib.MultivariateNormal):
		return q.mean
	elif isinstance(q, distrib.Normal):
		return q.loc
	elif isinstance(q, distrib.Categorical):
		return q.logits.max(dim=-1)[1]
	return q.mle
