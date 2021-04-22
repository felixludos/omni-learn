
import sys, os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as distrib
from torch.distributions.utils import lazy_property

from omnibelt import InitWall
import omnifig as fig

#####################
# region Simple math
#####################

def factors(n): # has duplicates, starts from the extremes and ends with the middle
	return (x for tup in ([i, n//i]
				for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup)

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

def pairwise_distance(ps, qs=None, p=2): # last dim is summed

	if qs is None:
		return F.pdist(ps, p=p)

	ps, qs = ps.unsqueeze(-2), qs.unsqueeze(-3)
	return (ps - qs).pow(p).sum(-1).pow(1/p)

def lorentzian(ps, qs=None, C=None):

	dists = pairwise_distance(ps, qs).pow(2)

	if C is None:
		C = ps.size(-1) # dimensionality

	return C / (C + dists)

def MMD(p, q, C=None):

	if C is None:
		C = q.size(-1)

	ps = lorentzian(p, C=C)
	qs = lorentzian(q, C=C)
	pq = lorentzian(p, q, C=C)

	return ps.mean() + qs.mean() - 2*pq.mean()

# endregion
#####################
# region Neural Networks
#####################


def logify(x, eps=1e-15):
	high = x > 1.
	low = x < -1.
	ok = ~ (high + low)

	v = x.abs().add(eps).log().add(1)
	return ok * x + high * v - low * v

@fig.Component('logifier')
class Logifier(nn.Module):
	def __init__(self, eps=1e-15):
		super().__init__()
		self.eps = eps
	def forward(self, x):
		return logify(x, eps=self.eps)

def unlogify(x):
	high = x > 1.
	low = x < -1.
	ok = ~ (high + low)

	v = x.abs().sub(1).exp()
	return ok * x + high * v - low * v

@fig.Component('unlogifier')
class Unlogifier(nn.Module):
	def forward(self, x):
		return unlogify(x)


class RMSELoss(nn.MSELoss):
	def forward(self, *args, **kwargs):
		loss = super().forward(*args, **kwargs)
		return loss.sqrt()

@fig.AutoComponent('loss')
def get_loss_type(ident, **kwargs):

	if not isinstance(ident, str):
		return ident

	if ident == 'mse':
		return nn.MSELoss(**kwargs)
	elif ident == 'rmse':
		return RMSELoss(**kwargs)
	elif ident == 'l1':
		return nn.L1Loss(**kwargs)
	elif ident == 'huber':
		return nn.SmoothL1Loss(**kwargs)
	elif ident == 'nll':
		print('WARNING: should probably use cross-entropy')
		return nn.NLLLoss(**kwargs)
	elif ident == 'cross-entropy':
		return nn.CrossEntropyLoss(**kwargs)
	elif ident == 'kl-div':
		return nn.KLDivLoss(**kwargs)
	elif ident == 'bce':
		#print('WARNING: should probably use bce-log')
		return nn.BCELoss(**kwargs)
	elif ident == 'bce-log':
		return nn.BCEWithLogitsLoss(**kwargs)
	else:
		assert False, "Unknown loss type: " + ident


@fig.AutoComponent('regularization')
def get_regularization(ident, p=2, dim=1, reduction='mean'):

	if not isinstance(ident, str):
		return ident

	if ident == 'L2' or ident == 'l2':
		return Lp_Norm(p=2, dim=dim, reduction=reduction)
	elif ident == 'L1' or ident == 'l1':
		return Lp_Norm(p=1, dim=dim, reduction=reduction)
	elif ident == 'Lp':
		return Lp_Norm(p=p, dim=dim, reduction=reduction)
	elif ident == 'pow2':
		return lambda q: q.pow(2).sum()
	else:
		print(f'Unknown reg: {ident}')
		# raise Exception(f'unknown: {name}')



class Mish(nn.Module):
	def forward(self, x):
		return x * torch.tanh(F.softplus(x))
class Swish(nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)

# Choose non-linearities
@fig.AutoComponent('nonlin')
def get_nonlinearity(ident, dim=1, inplace=True, **kwargs):

	if ident is None:
		return None
	if not isinstance(ident, str):
		return ident

	if ident == 'prelu':
		return nn.PReLU(**kwargs)
	elif ident == 'lrelu':
		return nn.LeakyReLU(**kwargs)
	elif ident == 'relu':
		return nn.ReLU(inplace=inplace)
	elif ident == 'tanh':
		return nn.Tanh()
	elif ident == 'log-softmax':
		return nn.LogSoftmax(dim=dim)
	elif ident == 'softmax':
		return nn.Softmax(dim=dim)
	elif ident == 'softmax2d':
		return nn.Softmax2d()
	elif ident == 'softplus':
		return nn.Softplus(**kwargs)
	elif ident == 'sigmoid':
		return nn.Sigmoid()
	elif ident == 'elu':
		return nn.ELU(inplace=inplace, **kwargs)
	elif ident == 'selu':
		return nn.SELU(inplace=inplace, **kwargs)

	elif ident == 'mish':
		return Mish()
	elif ident == 'swish':
		return Swish()

	else:
		assert False, "Unknown nonlin type: " + ident

@fig.AutoComponent('lp-norm')
class Lp_Normalization(nn.Module):
	def __init__(self, p=2, dim=1, eps=1e-8):
		super().__init__()
		self.p = p
		self.dim = dim
		self.eps = eps

	def extra_repr(self):
		return 'p={}'.format(self.p)

	def forward(self, x):
		return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)

class Lp_Norm(nn.Module):
	def __init__(self, p=2, dim=None, reduction='mean'):
		super().__init__()
		self.p = p
		self.dim = dim
		self.reduction = reduction

	def extra_repr(self):
		return 'p={}'.format(self.p)

	def forward(self, x):

		mag = x.norm(p=self.p, dim=self.dim)

		if self.dim is None:
			return mag

		if self.reduction == 'mean':
			return mag.mean()
		elif self.reduction == 'sum':
			return mag.sum()
		else:
			return mag

@fig.AutoComponent('normalization')
def get_normalization(ident, channels, groups=8, p=2, **kwargs):

	if not isinstance(ident, str):
		return ident

	if ident == 'batch':
		return nn.BatchNorm2d(channels, **kwargs)
	if ident == 'instance':
		return nn.InstanceNorm2d(channels, **kwargs)
	if ident == 'l1':
		return Lp_Normalization(1)
	if ident == 'l2':
		return Lp_Normalization(2)
	if ident == 'lp':
		return Lp_Normalization(p=p, **kwargs)
	if ident == 'group':
		return nn.GroupNorm(groups, channels, **kwargs)
	raise Exception(f'unknown norm type: {ident}')

@fig.AutoComponent('down-pooling')
def get_pooling(ident, down, chn=None):
	if not isinstance(ident, str):
		return ident

	if down == 1:
		return None

	if ident == 'conv':
		assert chn is not None
		return nn.Conv2d(chn, chn, kernel_size=down, padding=0, stride=down)
	elif ident == 'max':
		return nn.MaxPool2d(down, down)
	elif ident == 'avg':
		return nn.AvgPool2d(down, down)

	raise Exception(f'unknown pool type: {ident}')

@fig.AutoComponent('up-pooling')
def get_upsample(ident, up=2, size=None, channels=None):
	if not isinstance(ident, str):
		return ident

	if up == 1:
		return None

	if ident == 'conv':
		assert channels is not None
		return nn.ConvTranspose2d(channels, channels, kernel_size=up, stride=up)
	else:
		assert up is not None or size is not None
		if size is not None:
			return nn.Upsample(size=size, mode=ident)
		return nn.Upsample(scale_factor=up, mode=ident)


def conv_size_change(H, W, kernel_size=(3,3), padding=(1,1), stride=(1,1), dilation=(1,1)):
	H = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
	W = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
	return H, W

def deconv_size_change(H, W, kernel_size=(4,4), padding=(1,1), stride=(1,1), dilation=(1,1), output_padding=(0,0)):
	H = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
	W = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
	return H, W



# endregion
#####################
# region Randomness and Noise
#####################

def set_seed(seed=None):
	if seed is None:
		seed = gen_random_seed()
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	return seed

def gen_random_seed():
	return random.getrandbits(32)

def gen_deterministic_seed(seed):
	set_seed(seed)
	return gen_random_seed()

def subset(seq, k=None, seed=None):
	'''
	Create a reproducable subset of k samples (using seed), without changing the global random state
	(unless seed is not provided).

	seq can be an int: seq <- [0...seq-1]
	'''
	if seed is not None:
		rng = np.random.RandomState()
		rng.seed(seed)
	else:
		rng = np.random

	perm = rng.permutation(seq)
	if k is not None:
		return perm[:k]
	return perm


class OUNoise(nn.Module):
	"""docstring for OUNoise"""
	def __init__(self, dim=1, batch_size=None,
				 mu=0., theta=1., sigma=1.,
				 batch_first=True):
		super(OUNoise, self).__init__()

		if not isinstance(mu, torch.Tensor):
			mu = torch.tensor(mu)
		if not isinstance(theta, torch.Tensor):
			theta = torch.tensor(theta)
		if not isinstance(sigma, torch.Tensor):
			sigma = torch.tensor(sigma)

		mu = mu.float()
		sigma = sigma.float()
		theta = theta.float()

		if mu.ndimension() > 0:
			dim = mu.size(-1)
			if mu.ndimension() > 1:
				batch_size = mu.size(0)
		elif sigma.ndimension() > 0:
			dim = sigma.size(-1)
			if sigma.ndimension() > 1:
				batch_size = sigma.size(0)
		elif theta.ndimension() > 0:
			dim = theta.size(-1)
			if theta.ndimension() > 1:
				batch_size = theta.size(0)

		if mu.ndimension() == 0:
			mu = mu.unsqueeze(-1).expand(dim)
		if sigma.ndimension() == 0:
			sigma = sigma.unsqueeze(-1).expand(dim)
		if theta.ndimension() == 0:
			theta = theta.unsqueeze(-1).expand(dim)

		assert mu.size(-1) == dim and sigma.size(-1) == dim and theta.size(-1) == dim, 'wrong dims'

		self.register_buffer('mu', mu)
		self.register_buffer('sigma', sigma)
		self.register_buffer('theta', theta)
		self.register_buffer('state', torch.tensor(0.)) # register placeholder

		self.dim = dim
		self.batch_size = None
		self.batch_first = batch_first
		self.reset(batch_size)

	def reset(self, B=None):
		if B is not None:
			self.batch_size = B
		self.stack_dim = 1 if self.batch_first and self.batch_size is not None else 0

		self.state = self.mu.clone()

		if self.batch_size is not None:
			if self.state.ndimension() < 2:
				self.state = self.state.unsqueeze(0).expand((self.batch_size, self.dim))
			assert self.state.size(0) == self.batch_size, '{} vs {}'.format(self.state.size(0), self.batch_size)

		return self.state

	def forward(self, seq=None, reset=False):
		if reset:
			self.reset()
		nseq = 1 if seq is None else seq

		states = [self.state]
		for k in range(nseq):
			dx = self.theta * (self.mu - states[-1]) + self.sigma * torch.randn_like(self.state)
			states.append(states[-1] + dx)

		self.state = states[-1]
		states = torch.stack(states[1:], self.stack_dim) # batch x seq x dim or seq x batch x dim

		if seq is None:
			states = states.squeeze(self.stack_dim)
		if self.batch_size is None and states.ndimension() > 1:
			states.squeeze(1-self.stack_dim)

		return states

def random_permutation_mat(D):
	return torch.eye(D)[torch.randperm(D)]

def shuffle_dim(M, dim=-1): # reshuffle with replacement
	assert M.ndimension() == 2 and dim == -1, 'not implemented yet' # TODO: generalize to many-dimmed M and any dim
	B,D = M.size()
	
	row = torch.randperm(B*D).fmod(B).view(B, D)
	col = torch.arange(D).unsqueeze(0).expand(B,-1)
	
	return M[row, col]

# endregion
#####################
# region Linear Systems
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

def orthogonalize(M): # gram schmidt process
	def projection(u, v):
		return (v * u).sum() / (u * u).sum() * u

	nk = M.size(0)
	uu = torch.zeros_like(M, device=M.device)
	uu[:, 0] = M[:, 0].clone()
	for k in range(1, nk):
		vk = M[k].clone()
		uk = 0
		for j in range(0, k):
			uj = uu[:, j].clone()
			uk = uk + projection(uj, vk)
		uu[:, k] = vk - uk
	for k in range(nk):
		uk = uu[:, k].clone()
		uu[:, k] = uk / uk.norm()
	return uu

# def solve_reg(A, b, out=None, bias=True, l2_reg=0):
#
# 	R = A.t() @ A


# endregion
#####################
# region Affine transformations/conversions
#####################

def rots_2d(thetas):
	thetas = thetas.view(-1)

	sin = torch.sin(thetas)
	cos = torch.cos(thetas)

	return torch.stack([
		torch.stack([cos, -sin], -1),
		torch.stack([sin, cos], -1)
	], -2)


def se2_tfm(R, t=None):
	if t is None:
		shape = R.shape[:-1]
		t = torch.zeros(*shape, 1, device=R.device)
	return torch.cat([R, t], -1)



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



# endregion
#####################
# region Angles/Spheres
#####################

def cart2angl(pts):
	N, D = pts.size()

	if D == 2:
		phis = torch.atan2(pts.narrow(-1, 1, 1), pts.narrow(-1, 0, 1))
	else:
		num = pts.narrow(-1, 0, D - 1)
		last = pts.narrow(-1, D - 1, 1)

		den = torch.triu(pts.unsqueeze(1).expand(N,D,D)).pow(2).sum(-1).sqrt().narrow(-1, 0, D - 1)

		phis = torch.acos(num / den)

		sel = last.squeeze() < 0
		phis[sel, -1] = 2 * np.pi - phis[sel, -1]

	return phis

def cart2sphr(pts):

	N, D = pts.size()

	assert D >= 2

	r = pts.norm(dim=-1, keepdim=True)
	phis = cart2angl(pts)

	return torch.cat([r, phis],-1)

def angl2cart(phis):
	N, S = phis.size()

	cos = torch.cos(phis)
	sin = torch.sin(phis)

	sel = torch.tril(torch.ones(S, S, device=phis.device))

	sns = sin.unsqueeze(1).pow(sel.unsqueeze(0)).prod(-1)

	first = cos.narrow(-1, 0, 1)
	last = sns.narrow(-1, -1, 1)
	middle = cos[:, 1:] * sns[:, :-1]

	return torch.cat([first, middle, last], -1)

def sphr2cart(sphr):
	N, D = sphr.size()

	r, phis = sphr.narrow(-1,0,1), sphr.narrow(-1,1,D-1)

	pts = angl2cart(phis)

	return r * pts




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


# endregion
#####################
# region Points
#####################

def pairwise_displacements(a):
	n = a.shape[0]
	d = a.shape[1]
	c = n * (n - 1) // 2
	
	out = np.zeros((c, d))
	
	l = 0
	r = l + n - 1
	for sl in range(1, n):  # no point1 - point1!
		out[l:r] = a[:n - sl] - a[sl:]
		l = r
		r += n - (sl + 1)
	return out


def pairwise_displacements_2(a):
	n = a.shape[0]
	d = a.shape[1]
	c = n * (n - 1) // 2
	
	out = []
	
	l = 0
	r = l + n - 1
	for sl in range(1, n):  # no point1 - point1!
		out.append(a[:n - sl] - a[sl:])
		l = r
		r += n - (sl + 1)
	return np.concatenate(out)


# endregion
#####################
# region Probabilities/Statistics
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


class Normal_Mixture(distrib.Distribution):
	def __init__(self, locs, scales, wts=None, batch_first=False):

		super().__init__(locs.size()[1:], locs.size()[1:])

		self.locs = locs
		self.scales = scales
		self.denom = scales.pow(2).mul(2)

		self.dim = int(batch_first)

		if wts is None:
			# wts = torch.ones(self.locs.size(0), device=locs.device)
			pass
		else:
			raise NotImplementedError
			assert wts.size(0) == self.locs.size(self.dim)
		self.wts = wts

	@property
	def mean(self):
		if self.wts is None:
			return self.locs.mean(self.dim)

	@property
	def stddev(self):
		if self.wts is None:
			return self.scales

	def log_prob(self, value):
		value = value.unsqueeze(self.dim)
		log_probs = (self.locs - value).pow(2).div(-self.denom) - self.denom.mul(np.pi).log().div(2)
		return log_probs.logsumexp(log_probs, dim=self.dim)

	def rsample(self, sample_shape=None):

		if self.wts is None:



			pass
		else:
			raise NotImplementedError

		pass


_known_distribs = [distrib.MultivariateNormal, distrib.Categorical, Joint_Distribution]

def group_kl(p, q):
	return sum([distrib.kl.kl_divergence(p_i, q_i) for p_i, q_i in zip(p.base, q.base)])
distrib.kl.register_kl(Joint_Distribution, Joint_Distribution)(group_kl)

def standard_kl(p, q=None):
	mu, sigma = p.loc, p.scale
	return (mu.pow(2) - sigma.clamp(min=1e-20).log() + sigma - 1) / 2
distrib.kl.register_kl(distrib.Normal, type(None))(standard_kl)


def MLE(q):
	if isinstance(q, distrib.MultivariateNormal):
		return q.mean
	elif isinstance(q, distrib.Normal):
		return q.loc
	elif isinstance(q, distrib.Categorical):
		return q.logits.max(dim=-1)[1]
	return q.mle

# endregion
#####################
