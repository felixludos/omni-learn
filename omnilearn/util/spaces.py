
from omnibelt import unspecified_argument
import omnifig as fig

import numpy as np
import torch
from torch.nn import functional as F

from .math import angle_diff, gen_deterministic_seed
from .features import DeviceBase, Configurable

# TODO: include dtypes


class DimSpec:
	def __init__(self, min=None, max=None, shape=(1,), **kwargs):
		
		if isinstance(shape, int):
			shape = (shape,)
		
		super().__init__(**kwargs)
		
		self._shape = shape
		
		if min is not None:
			min = torch.as_tensor(min).float().expand(shape)
		self._min = min
		
		if max is not None:
			max = torch.as_tensor(max).float().expand(shape)
		self._max = max


	def __str__(self):
		return f'{self.__class__.__name__}'


	def __repr__(self):
		return str(self)

		
	def compress(self, vals):
		return vals
	
	
	def expand(self, vals):
		return vals
	
	
	@property
	def shape(self):
		return self._shape

	
	@property
	def expanded_shape(self):
		return self.shape
	
	
	def __len__(self): # compressed shape
		return np.product(self.shape).item()
	
	
	def expanded_len(self):
		return np.product(self.expanded_shape).item()
	
	
	def sample(self, N=None, gen=None, seed=None): # samples compressed
		raise NotImplementedError
	
	
	@property
	def min(self):
		return self._min
	
	
	@property
	def max(self):
		return self._max
		
		
	@property
	def range(self):
		return self.max - self.min
	
	
	def transform(self, vals, spec=None):
		if spec is not None:
			vals = spec.standardize(vals)
		return self.unstandardize(vals)


	def difference(self, x, y, standardize=False): # x-y
		x = self.standardize(x)
		y = self.standardize(y)
		return x-y


	def distance(self, x, y, standardize=False):
		return self.difference(x,y, standardize=standardize).pow(2).sum(-1).sqrt()

	
	def standardize(self, vals):
		raise NotImplementedError
	
	
	def unstandardize(self, vals):
		raise NotImplementedError
	
	

class DenseDim(DimSpec):
	
	def sample(self, N=None, gen=None, seed=None):

		if seed is not None:
			gen = torch.Generator()
			gen.manual_seed(seed)

		# kwargs = {} if gen is None else {'gen': gen}

		sqz = N is None
		if N is None:
			N = 1

		samples = self.unstandardize(self._sample((N, *self.shape), gen))
		return samples.squeeze(0) if sqz else samples


	def _sample(self, shape, generator):
		return torch.randn(*shape, generator=generator)



class HalfBoundDim(DenseDim):
	def __init__(self, bound=0, side='lower', bound_type='exp', epsilon=1e-10,
	             min=None, max=None, **kwargs):
		
		if min is None and max is None:
			assert bound is not None, 'No bound provided'
			assert side in {'lower', 'upper'}, 'Bound side not specified'
		
			if side == 'lower':
				min = bound
			else:
				max = bound
				
		assert max is None or min is None, f'Too many bounds specified: min={min}, max={max}'
		assert bound_type in {'exp', 'soft', 'chi', 'abs'}
		if bound_type in {'chi', 'abs'}:
			print(f'WARNING: half-bound dim using transformation {bound_type} cannot be standardized')
		super().__init__(max=max, min=min, **kwargs)
		
		self._bound_type = bound_type
		self._epsilon = epsilon


	def __str__(self):
		lim = f'min={self.min.mean().item():.3g}' if self.min is not None else f'max={self.max.mean().item():.3g}'
		return f'HalfBound({lim})'


	def standardize(self, vals):
		if self.min is not None:
			vals = vals - self.min.unsqueeze(0)
		elif self.max is not None:
			vals = vals.sub(self.max.unsqueeze(0)).mul(-1)

		vals = vals.clamp(min=self._epsilon)
		if self._bound_type == 'soft':
			vals = vals.exp().sub(1).log()
		elif self._bound_type == 'chi':
			vals = vals.sqrt()
		elif self._bound_type == 'exp':
			vals = vals.log()

		return vals


	def unstandardize(self, vals):
		if self._bound_type == 'soft':
			vals = F.softplus(vals)
		elif self._bound_type == 'chi':
			vals = vals.pow(2)
		elif self._bound_type == 'exp':
			vals = vals.exp()
		else:
			vals = vals.abs()

		if self.min is not None:
			vals = vals + self.min.unsqueeze(0)
		elif self.max is not None:
			vals = -vals + self.max.unsqueeze(0)

		return vals



class BoundDim(DenseDim):
	def __init__(self, min=0., max=1., epsilon=1e-10, **kwargs):
		super().__init__(min=min, max=max, **kwargs)
		assert self.min is not None, f'No lower bound provided'
		assert self.max is not None, f'No upper bound provided'

		self._epsilon = epsilon


	def __str__(self):
		return f'Bound(min={self.min.mean().item():.3g}, max={self.max.mean().item():.3g})'


	def _sample(self, shape, generator):
		return torch.rand(*shape, generator=generator)


	def standardize(self, vals):
		return vals.sub(self.min.unsqueeze(0)).div(self.range.unsqueeze(0))\
			.clamp(min=self._epsilon, max=1-self._epsilon)
	
	
	def unstandardize(self, vals):
		return vals.clamp(min=self._epsilon, max=1-self._epsilon)\
			.mul(self.range.unsqueeze(0)).add(self.min.unsqueeze(0))


	def difference(self, x, y, standardize=False):
		return super().difference(x, y, standardize=standardize) * (self.range.unsqueeze(0) ** float(not standardize))



class UnboundDim(DenseDim):
	def __init__(self, min=None, max=None, **kwargs):
		super().__init__(min=None, max=None, **kwargs)

	def __str__(self):
		return 'Unbound()'

	def standardize(self, vals):
		return vals


	def unstandardize(self, vals):
		return vals



class PeriodicDim(BoundDim):
	def __init__(self, period=1., min=0., max=None, **kwargs):
		assert min is not None and (period is not None or max is not None), 'Not enough bounds provided'
		if max is None:
			max = min + period
		super().__init__(min=min, max=max, **kwargs)


	def __str__(self):
		return f'Periodic({self.period.mean().item():.3g})'

	
	@property
	def period(self):
		return self.range
		
		
	@property
	def expanded_shape(self):
		return (*self.shape, 2)
	
	
	def expand(self, vals):
		thetas = vals.view(-1, *self.shape).sub(self.min).mul(2*np.pi/self.period)
		return torch.stack([thetas.cos(), thetas.sin()], -1)
	
	
	def compress(self, vals):
		vals = vals.view(-1, *self.expanded_shape)
		return torch.atan2(vals[...,1], vals[...,0]).div(2*np.pi/self.period).remainder(self.period).add(self.min)


	def difference(self, x, y, standardize=False):
		return angle_diff(self.standardize(x), self.standardize(y), period=1.) * (self.period ** float(not standardize))


	def transform(self, vals, spec):
		if isinstance(spec, CategoricalDim):
			spec.n += 1
		out = super().transform(vals, spec)
		if isinstance(spec, CategoricalDim):
			spec.n -= 1
		return out



class MultiDimSpace(DimSpec):
	def __init__(self, channels, shape=None, channel_first=False, **kwargs):

		if isinstance(shape, int):
			shape = (shape,)

		size = shape
		shape = (channels,) if shape is None else (
			(channels, *shape) if channel_first else (*shape, channels))

		super().__init__(shape=shape, **kwargs)
		self.channel_first = channel_first
		self.size = size
		self.channels = channels



class SimplexSpace(MultiDimSpace, BoundDim):

	def sample(self, N=None, **kwargs): # from Donald B. Rubin, The Bayesian bootstrap Ann. Statist. 9, 1981, 130-134.
		# discussed in https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex

		sqz = False
		if N is None:
			sqz = True
			N = 1

		extra_shape = [N, *self.shape]
		raw = super().sample(N=N, **kwargs).view(*extra_shape)

		dim = (-1) ** (not self.channel_first)
		extra_shape[dim] = 1

		edges = torch.cat([torch.zeros(*extra_shape),
		                   raw.narrow(dim, 0, self.channels - 1).sort(dim)[0],
		                   torch.ones(*extra_shape)], dim)

		samples = edges.narrow(dim, 1, self.channels) - edges.narrow(dim, 0, self.channels)
		return samples.squeeze(0) if sqz else samples

	def standardize(self, vals):
		return F.normalize(vals, p=1, dim=(-1) ** (not self.channel_first))

	def unstandardize(self, vals):
		return self.standardize(vals)



class SphericalSpace(MultiDimSpace, UnboundDim):

	def standardize(self, vals):
		return F.normalize(vals, p=2,
		                   dim=(-1) ** (not self.channel_first),  # 1 or -1
		                   )


	def unstandardize(self, vals):
		return self.standardize(vals)


	def euclidean_difference(self, x, y, standardize=False):
		return super().difference(x, y, standardize=False)


	def geodesic_difference(self, x, y, standardize=False):
		raise NotImplementedError


	def difference(self, x, y, standardize=False): # geodesic by default
		return self.geodesic_difference(x, y, standardize=False)



class SpatialSpace(MultiDimSpace):
	def __init__(self, channels, size, **kwargs):
		super().__init__(shape=shape, **kwargs)



class SequenceSpace(SpatialSpace):
	def __init__(self, channels=1, length=None, **kwargs):
		super().__init__(channels=channels, size=(length,), channel_first=True, **kwargs)
		self.length = length



class ImageSpace(SpatialSpace):
	def __init__(self, channels=1, height=None, width=None, **kwargs):
		super().__init__(channels=channels, size=(height, width), **kwargs)
		self.height = height
		self.width = width



class VolumeSpace(SpatialSpace):
	def __init__(self, channels=1, height=None, width=None, depth=None, **kwargs):
		super().__init__(channels=channels, size=(height, width, depth), **kwargs)
		self.height = height
		self.width = width
		self.depth = depth



class CategoricalDim(DimSpec):
	def __init__(self, n, **kwargs):
		super().__init__(min=0, max=n - 1, **kwargs)
		self._min = self._min.long()
		self._max = self._max.long()
		self.n = n


	def __str__(self):
		return f'Categorical({self.n})'


	def standardize(self, vals):
		return vals/(self.n-1)
	
	
	def unstandardize(self, vals):
		return (vals * self.n).long().clamp(max=self.n-1)
	
	
	@property
	def expanded_shape(self):
		return (*self.shape, self.n)
	
	
	def sample(self, N=None, gen=None, seed=None):
		if seed is not None:
			gen = torch.Generator()
			gen.manual_seed(seed)
		
		# kwargs = {} if gen is None else {'gen':gen}
		
		sqz = N is None
		if N is None:
			N = 1
		
		samples = torch.randint(self.n, size=(N, *self.shape), generator=gen)
		
		return samples.squeeze(0) if sqz else samples
		
		
	def expand(self, vals):
		return F.one_hot(vals.long(), self.n)
		
		
	def compress(self, vals):
		return vals.argmax(-1)


	def difference(self, x, y, standardize=False):
		return x.sub(y).bool().long()


# class CategoricalProbsDim(CategoricalDim):
#
# 	def difference(self, x, y):
# 		return F.binary_cross_entropy(x, y)
#
# class CategoricalLogitsDim(CategoricalProbsDim):
#
#
#
# 	def difference(self, x, y):
# 		return F.binary_cross_entropy_with_logits(x, y)


class BinaryDim(CategoricalDim):
	def __init__(self, n=None, **kwargs):
		super().__init__(n=2, **kwargs)

	def __str__(self):
		return f'Binary()'



class JointSpace(DimSpec):
	def __init__(self, *dims, shape=None, max=None, min=None, **kwargs):
		singles = []
		for d in dims:
			if isinstance(d, JointSpace):
				singles.extend(d.dims)
			else:
				singles.append(d)
		dims = singles
		shape = (sum(len(dim) for dim in dims),)
		expanded_shape = sum(dim.expanded_len() for dim in dims)
		
		super().__init__(shape=shape, min=None, max=None, **kwargs)
		
		self._expanded_shape = expanded_shape
		self.dims = dims
		self._is_dense = any(1 for dim in dims if isinstance(dim, DenseDim))


	def __str__(self):
		contents = ', '.join(str(x) for x in self.dims)
		return f'Joint({contents})'


	def __iter__(self):
		return iter(self.dims)


	@property
	def expanded_shape(self):
		return self._expanded_shape
	
	
	def _dispatch(self, method, *vals, use_expanded=False, split_kwargs=[], **base_kwargs):
		
		outs = []
		idx = 0
		B = None
		for i, dim in enumerate(self.dims):
			D = dim.expanded_len() if use_expanded else len(dim)
			args = tuple((v.narrow(-1, idx, D) if isinstance(v, torch.Tensor) else v[i]) for v in vals)
			kwargs = base_kwargs.copy()
			for key in split_kwargs:
				kwargs[key] = kwargs[key][i]
			
			out = getattr(dim, method)(*args, **kwargs)
			if B is None:
				B = out.size(0) if len(out.size()) > 1 else 1
			out = out.view(B, -1)
			if self._is_dense:
				out = out.float()
			outs.append(out)
			
			idx += D
			
		return torch.cat(outs, -1)
	
	
	def standardize(self, vals):
		return self._dispatch('standardize', vals)
	
	
	def unstandardize(self, vals):
		return self._dispatch('unstandardize', vals)
	
	
	def expand(self, vals):
		return self._dispatch('expand', vals)
	
	
	def compress(self, vals):
		return self._dispatch('compress', vals, use_expanded=True)
	
	
	def sample(self, N=None, gen=None, seed=None):
		assert gen is None # TODO
		if seed is not None:
			seeds = [seed % (2**32-1)]
			for _ in range(len(self)-1):
				seeds.append(gen_deterministic_seed(seeds[-1]))
			return self._dispatch('sample', split_kwargs=['seed'], N=N, seed=seeds)
		return self._dispatch('sample', N=N, gen=gen, seed=seed)


	def difference(self, x, y, standardize=False):
		return self._dispatch('difference', x, y, standardize=standardize)


	def __getitem__(self, item):
		return self.dims[item]



class DimSpecC(Configurable, DimSpec):
	def __init__(self, A, min=unspecified_argument, max=unspecified_argument,
	             shape=unspecified_argument, **kwargs):
		
		if min is unspecified_argument:
			min = A.pull('min', None)
		if max is unspecified_argument:
			max = A.pull('max', None)
			
		if shape is unspecified_argument:
			shape = A.pull('shape', (1,))
		
		super().__init__(A, min=min, max=max, shape=shape, **kwargs)



@fig.Component('space/half-bound')
class HalfBoundDimC(DimSpecC, HalfBoundDim):
	def __init__(self, A, bound=unspecified_argument, side=unspecified_argument,
	             bound_type=unspecified_argument, epsilon=unspecified_argument, **kwargs):
		
		if bound is unspecified_argument:
			bound = A.pull('bound', 0.)
		if side is unspecified_argument:
			side = A.pull('side', 'lower')
		if bound_type is unspecified_argument:
			bound_type = A.pull('sample-type', 'exp')
		if epsilon is unspecified_argument:
			epsilon = A.pull('epsilon', 1e-10)

		super().__init__(A, bound=bound, side=side,
		                 bound_type=bound_type, epsilon=epsilon, **kwargs)



@fig.Component('space/bound')
class BoundDimC(DimSpecC, BoundDim):
	def __init__(self, A, epsilon=unspecified_argument, **kwargs):
		if epsilon is unspecified_argument:
			epsilon = A.pull('epsilon', 1e-10)
		super().__init__(A, epsilon=epsilon, **kwargs)



@fig.Component('space/unbound')
class UnboundDimC(DimSpecC, UnboundDim):
	pass



@fig.Component('space/periodic')
class PeriodicDimC(DimSpecC, PeriodicDim):
	def __init__(self, A, period=unspecified_argument, **kwargs):
		if period is unspecified_argument:
			period = A.pull('period', 1.)
			
		super().__init__(A, period=period, **kwargs)



@fig.Component('space/categorical')
class CategoricalDimC(DimSpecC, CategoricalDim):
	def __init__(self, A, n=unspecified_argument, **kwargs):
		if n is unspecified_argument:
			n = A.pull('n')
			
		super().__init__(A, n=n, **kwargs)



@fig.Component('space/joint')
class JointSpaceC(DimSpecC, JointSpace):
	def __init__(self, A, dims=unspecified_argument, **kwargs):
		if dims is unspecified_argument:
			dims = A.pull('dims')
		
		super().__init__(A, _req_args=dims)
	




