
import sys, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import framework as fm
from .. import util
from .atom import *

# deep network of 1D convs with kernel=1
def make_pointnet(input_dim, output_dim, hidden_dims=[],
                  batch_norm=False, output_batch_norm=False,
                  nonlin='prelu', output_nonlin=None):
	nonlins = [nonlin] * len(hidden_dims) + [output_nonlin]
	bns = [batch_norm] * len(hidden_dims) + [output_batch_norm]
	hidden_dims = [input_dim] + hidden_dims + [output_dim]

	layers = []
	for in_dim, out_dim, nonlin, bn in zip(hidden_dims, hidden_dims[1:], nonlins, bns):
		layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
		if bn:
			layers.append(nn.BatchNorm1d(out_dim))
		if nonlin is not None:
			layers.append(util.get_nonlinearity(nonlin))

	return nn.Sequential(*layers)


_point_str_tmp = '{: >4} -> {: >12} -> {: >4}'
_point_str_pool_tmp = '>> {} <<'

class PointPooling(nn.Module):
	def __init__(self,):
		super().__init__()

	def point_str(self):
		proc = '[unknown]'
		return _point_str_pool_tmp.format(proc)

class PointMaxPooling(PointPooling):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.max(-1)[0]

	def point_str(self):
		proc = 'max'
		return _point_str_pool_tmp.format(proc)


class PointAvgPooling(PointPooling):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.mean(-1)

	def point_str(self):
		proc = 'avg'
		return _point_str_pool_tmp.format(proc)

class PointStdPooling(PointPooling):
	def __init__(self,):
		super().__init__()

	def forward(self, x):
		return x.std(-1)

	def point_str(self):
		proc = 'std'
		return _point_str_pool_tmp.format(proc)

class PointWeightedAvgPooling(PointPooling):
	def __init__(self, din, hidden_dims=[], nonlin='prelu', batch_norm=False):
		super().__init__()
		self.din = din

		self.net = make_pointnet(din, 1, hidden_dims=hidden_dims, nonlin=nonlin, batch_norm=batch_norm)
		self.norm = util.get_nonlinearity('softmax', dim=-1)

	def forward(self, x):
		w = self.norm(self.net(x))
		return (w*x).mean(-1)

	def point_str(self):
		proc = 'wt'
		return _point_str_pool_tmp.format(proc)

def get_point_pooling(pooling, **kwargs):
	if pooling == 'avg':
		return PointAvgPooling()
	elif pooling == 'max':
		return PointMaxPooling()
	elif pooling == 'std':
		return PointStdPooling()
	elif pooling == 'wavg':
		return PointWeightedAvgPooling(**kwargs)
	return None


class PointTransform(fm.Model):
	def __init__(self, din, dout):
		super().__init__(din, dout)

	def point_str(self):
		proc = '[unknown]'
		return _point_str_tmp.format(self.din, proc, self.dout)

class PointSelfTransform(PointTransform):
	def __init__(self, din, dout=None, hidden_dims=[], nonlin='prelu', batch_norm=False):
		if dout is None:
			dout = din
		super().__init__(din, dout)

		self.transform = make_pointnet(self.din, self.dout, hidden_dims=hidden_dims,
											nonlin=nonlin, batch_norm=batch_norm)

		self.hidden_dims = hidden_dims

	def forward(self, x):
		return self.transform(x)

	def point_str(self):
		proc = str(self.hidden_dims)
		return _point_str_tmp.format(self.din, proc, self.dout)

class PointMatrixTransform(PointTransform):
	def __init__(self, din, pool, dout=None, latent_dim=None, init_identity=True,
				 enc_hidden=[], dec_hidden=[], nonlin='prelu', batch_norm=False):
		if dout is None: # default is square
			dout = din
		super().__init__(din, dout)

		if latent_dim is None:
			dec = None
			latent_dim = din*dout
		else:
			dec = make_MLP(latent_dim, din*dout, hidden_dims=dec_hidden, nonlin=nonlin)

		self.enc = make_pointnet(din, latent_dim, hidden_dims=enc_hidden, nonlin=nonlin, batch_norm=batch_norm,
									  output_nonlin=(None if dec is None else nonlin),
									  output_batch_norm=(None if dec is None else batch_norm))
		self.pool = pool
		assert isinstance(self.pool, PointPooling), 'invalid pooling'
		self.dec = dec

		if self.dec is not None and init_identity:
			self.dec[-1].bias.data.view(dout, din).add_(torch.eye(dout, din))

	def forward(self, x): # single point cloud

		q = self.enc(x)
		q = self.pool(q)

		if self.dec is None:
			T = q.view(-1, self.dout, self.din)
		else:
			T = self.dec(q).view(-1, self.dout, self.din)

		return T @ x

	def point_str(self):
		proc = '({}, {})'.format(self.din, self.dout)
		return _point_str_tmp.format(self.din, proc, self.dout)


class PointCatTransform(PointTransform):
	def __init__(self, din, pool, catdim, latent_dim=None,
				 enc_hidden=[], dec_hidden=[], nonlin='prelu', batch_norm=False):
		super().__init__(din, din+catdim)

		self.catdim = catdim

		if latent_dim is None:
			dec = None
			latent_dim = self.catdim
		else:
			dec = make_MLP(latent_dim, self.catdim, hidden_dims=dec_hidden, nonlin=nonlin)

		self.enc = make_pointnet(din, latent_dim, hidden_dims=enc_hidden, nonlin=nonlin, batch_norm=batch_norm,
									  output_nonlin=(None if dec is None else nonlin),
									  output_batch_norm=(None if dec is None else batch_norm))
		self.pool = pool
		assert isinstance(self.pool, PointPooling), 'invalid pooling'
		self.dec = dec

	def forward(self, x):  # single point cloud

		q = self.enc(x)
		q = self.pool(q)

		if self.dec is not None:
			q = self.dec(q)

		q = q.unsqueeze(-1).expand(x.size(0), self.catdim, x.size(-1))

		return torch.cat([x, q], 1)

	def point_str(self):
		proc = '(+ {})'.format(self.catdim)
		return _point_str_tmp.format(self.din, proc, self.dout)

class PointMultiCatTransform(PointTransform):
	def __init__(self, din, catdim, ):
		pass


class PointCenterTransform(PointTransform):
	def __init__(self, din, num_centers, compose=True, learnable=True, dist_type=2):
		dout = num_centers
		if compose:
			dout += din
		super().__init__(din, dout)

		self.compose = compose

		centers = torch.randn(1, din, num_centers, 1)

		if learnable:
			self.centers = nn.Parameter(centers, requires_grad=True)
		else:
			self.register_buffer('centers', centers)

		self.dist_type = dist_type
		self.dist_fn = None
		if dist_type == 'cos':
			self.dist_fn = nn.CosineSimilarity()
		elif isinstance(dist_type, int):
			self.dist_fn = nn.PairwiseDistance(p=dist_type)
		else:
			raise Exception('invalid dist_type: {}'.format(dist_type))


	def forward(self, x):

		y = self.dist_fn(x.unsqueeze(2), self.centers)

		if self.compose:
			y = torch.cat([x, y], 1)

		return y

	def point_str(self):

		if isinstance(self.dist_type, int):
			dist = 'l{}'.format(self.dist_type)
		else:
			dist = self.dist_type

		comp = '+' if self.compose else '='

		proc = '{} {}{}'.format(dist, comp, self.centers.size(2))
		return _point_str_tmp.format(self.din, proc, self.dout)




class PointNet(fm.Model):
	def __init__(self, transforms, pool, din=None, output_dim=None,
				 hidden_dims=[], nonlin='prelu', output_nonlin=None):

		if len(transforms) and not isinstance(transforms, nn.ModuleList):
			transforms = nn.ModuleList(transforms)

		dout = din if din is not None else transforms[0].din
		for tfm in transforms:
			assert isinstance(tfm, PointTransform), '{} is not a valid transform'.format(tfm)
			assert tfm.din == dout, 'Invalid transform - dim dont match'
			dout = tfm.dout

		if output_dim is None:
			output_dim = dout
			net = None
		else:
			net = make_MLP(dout, output_dim, hidden_dims=hidden_dims, nonlin=nonlin)

		super().__init__(din, output_dim)

		self.transforms = transforms
		self.pool = pool
		self.hidden_dims = hidden_dims

		assert self.pool is not None or net is None, 'must pool when using net at the end (otherwise use a PointSelfTransform)'

		self.net = net
		self.nonlin = util.get_nonlinearity(output_nonlin) if output_nonlin is not None else None

	def forward(self, x):

		for tfm in self.transforms:
			x = tfm(x)

		q = self.pool(x) if self.pool is not None else x

		if self.net is not None:
			q = self.net(q)

		if self.nonlin is not None:
			q = self.nonlin(q)

		return q

	def point_str(self):
		tfm_strs = [tfm.point_str() for tfm in self.transforms]

		if self.pool is not None:
			tfm_strs.append(self.pool.point_str())

		if self.net is not None:
			tfm_strs.append('{} ==> {}'.format(str(self.hidden_dims), self.dout))

		return '\n'.join(tfm_strs)





# transforms.append(vision.PointSelfTransform(din=din, dout=4, hidden_dims=[8],
#                                      nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# transforms.append(vision.PointMatrixTransform(din=din, pool=vision.get_point_pooling('max'), dout=din, latent_dim=32,
#                                     enc_hidden=[8], dec_hidden=[16], nonlin=nonlin, batch_norm=batch_norm,
#                                     init_identity=True))
# din = transforms[-1].dout

# transforms.append(vision.PointSelfTransform(din=din, dout=6, hidden_dims=[8],
#                                      nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# transforms.append(vision.PointMatrixTransform(din=din, pool=vision.get_point_pooling('max'), dout=din, latent_dim=64,
#                                     enc_hidden=[], dec_hidden=[48,], nonlin=nonlin, batch_norm=batch_norm,
#                                     init_identity=True))
# din = transforms[-1].dout

# transforms.append(vision.PointCatTransform(din=din, pool=vision.get_point_pooling('max'), catdim=26, latent_dim=64,
#                                     enc_hidden=[32], dec_hidden=[32], nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# transforms.append(vision.PointSelfTransform(din=din, dout=64, hidden_dims=[48],
#                                      nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# transforms.append(vision.PointCenterTransform(din=din, num_centers=8, compose=False,
#                                               learnable=True, dist_type=2,
#                                              ))
# din = transforms[-1].dout

# din = 3

# transforms.append(vision.PointSelfTransform(din=din, dout=4, hidden_dims=[8],
#                                      nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout



# transforms.append(vision.PointCatTransform(din=din, pool=vision.get_point_pooling('max'), catdim=8, latent_dim=32,
#                                     enc_hidden=[8], dec_hidden=[16, 8], nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# transforms.append(vision.PointSelfTransform(din=din, dout=20, hidden_dims=[16],
#                                      nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# din = 3

# transforms.append(vision.PointCatTransform(din=din, pool=vision.get_point_pooling('max'), catdim=5, latent_dim=32,
#                                     enc_hidden=[8], dec_hidden=[16, 8], nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# transforms.append(vision.PointSelfTransform(din=din, dout=din, hidden_dims=[8],
#                                      nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# transforms.append(vision.PointCatTransform(din=din, pool=vision.get_point_pooling('max'), catdim=4, latent_dim=32,
#                                     enc_hidden=[8], dec_hidden=[16, 8], nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout

# transforms.append(vision.PointSelfTransform(din=din, dout=32, hidden_dims=[16],
#                                      nonlin=nonlin, batch_norm=batch_norm,
#                                     ))
# din = transforms[-1].dout







# Single point cloud - for unevenly sized point clouds




class SinglePointPooling(nn.Module):
	def __init__(self, dim=-2):
		super().__init__()
		self.dim = dim

class SinglePointMaxPooling(SinglePointPooling):
	def __init__(self, dim):
		super().__init__(dim)

	def forward(self, x):
		return x.max(self.dim)[0]


class SinglePointAvgPooling(SinglePointPooling):
	def __init__(self, dim):
		super().__init__(dim)

	def forward(self, x):
		return x.avg(self.dim)

class SinglePointStdPooling(SinglePointPooling):
	def __init__(self, dim):
		super().__init__(dim)

	def forward(self, x):
		return x.std(self.dim)

class SinglePointTransform(fm.Model):
	def __init__(self, din, dout):
		super().__init__(din, dout)

class SinglePointSelfTransform(SinglePointTransform):
	def __init__(self, din, dout=None, hidden_dims=[], nonlin='prelu'):
		if dout is None:
			dout = din
		super().__init__(din, dout)

		self.transform = make_MLP(self.din, self.dout, hidden_dims=hidden_dims, nonlin=nonlin)

	def forward(self, x):
		return self.transform(x)

class SinglePointMatrixTransform(SinglePointTransform):
	def __init__(self, din, dout=None, latent_dim=None, init_identity=True, pool_type='max',
				 enc_hidden=[], dec_hidden=[], nonlin='prelu'):
		if dout is None: # default is square
			dout = din
		super().__init__(din, dout)

		if latent_dim is None:
			dec = None
			latent_dim = din*dout
		else:
			dec = make_MLP(latent_dim, din*dout, hidden_dims=dec_hidden, nonlin=nonlin)

		self.enc = make_MLP(din, latent_dim, hidden_dims=enc_hidden, nonlin=nonlin, out_nonlin=(nonlin if dec is None else None))
		self.pool = get_point_pooling(pool_type)
		assert self.pool is not None, 'pooling cannot be None'
		self.dec = dec

		if self.dec is not None and init_identity:
			self.dec[-1].bias.view(din,dout).add_(torch.eye(din,dout))

	def forward(self, x): # single point cloud

		q = self.enc(x)
		q = self.pool(q)

		if self.dec is None:
			T = q.view(self.din, self.dout)
		else:
			T = self.dec(q).view(self.din, self.dout)

		print(x.shape, q.shape, T.shape)
		quit()

		return x @ T


class SinglePointCatTransform(SinglePointTransform):
	def __init__(self, din, catdim, latent_dim=None, pool_type='max',
				 enc_hidden=[], dec_hidden=[], nonlin='prelu'):
		super().__init__(din, din+catdim)

		self.catdim = catdim

		if latent_dim is None:
			dec = None
			latent_dim = self.catdim
		else:
			dec = make_MLP(latent_dim, self.catdim, hidden_dims=dec_hidden, nonlin=nonlin)

		self.enc = make_MLP(din, latent_dim, hidden_dims=enc_hidden, nonlin=nonlin,
								 out_nonlin=(nonlin if dec is None else None))
		self.pool = get_point_pooling(pool_type)
		assert self.pool is not None, 'pooling cannot be None'
		self.dec = dec

	def forward(self, x):  # single point cloud

		q = self.enc(x)
		q = self.pool(q)

		if self.dec is not None:
			q = self.dec(q)

		q = q.unsqueeze(0).expand(x.size(0))

		return torch.cat([x, q], -1)


class SinglePointNet(fm.Model):
	def __init__(self, transforms, pooling='max', din=3, hidden_dims=[], nonlin='prelu'):

		self.transforms = transforms
		assert len(self.transforms) > 0, 'not transforming anything'
		for tfm in transforms:
			assert isinstance(tfm, SinglePointTransform), '{} is not a valid transform'.format(tfm)

		super().__init__(din, self.transforms[-1].dout)

		self.pool = get_point_pooling(pooling)

		self.din = din


	def forward(self, xs): # can be a list of point clouds

		if isinstance(xs, torch.Tensor):
			xs = [xs]

		outs = []
		for x in xs:

			for tfm in self.transforms:
				x = tfm(x)

			outs.append(self.pool(x) if self.pool is not None else x)

		return torch.stack(outs) if self.pool is not None else outs





