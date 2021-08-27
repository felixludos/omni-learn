
from omnibelt import unspecified_argument
import omnifig as fig

import numpy as np
import torch
from torch.nn import functional as F

from .math import angle_diff
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
	
	
	def transform(self, vals, spec):
		return self.unstandardize(spec.standardize(vals))


	def difference(self, x, y): # x-y
		x = self.standardize(x)
		y = self.standardize(y)
		return x-y

	
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


	def _sample(self, shape, generator):
		return torch.rand(*shape, generator=generator)


	def standardize(self, vals):
		return vals.sub(self.min.unsqueeze(0)).div(self.range.unsqueeze(0))\
			.clamp(min=self._epsilon, max=1-self._epsilon)
	
	
	def unstandardize(self, vals):
		return vals.clamp(min=self._epsilon, max=1-self._epsilon)\
			.mul(self.range.unsqueeze(0)).add(self.min.unsqueeze(0))


	def difference(self, x, y):
		return super().difference(x, y) * self.range.unsqueeze(0)



class UnboundDim(DenseDim):
	def __init__(self, min=None, max=None, **kwargs):
		super().__init__(min=None, max=None, **kwargs)


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


	def difference(self, x, y):
		return angle_diff(self.standardize(x), self.standardize(y), period=1.) * self.period



class SpatialSpace(DimSpec):
	def __init__(self, channels, size, channel_first=False, **kwargs):
		shape = (channels, *size) if channel_first else (*size, channels)
		super().__init__(shape=shape, **kwargs)
		self.channel_first = channel_first
		self.img_size = size
		self.channels = channels



class SequenceSpace(SpatialSpace):
	def __init__(self, channels=1, length=None, **kwargs):
		super().__init__(channels=channels, size=(length,), **kwargs)
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
	def __init__(self, n, shape=(1,), **kwargs):
		super().__init__(min=0, max=n - 1, shape=shape, **kwargs)
		self._min = self._min.long()
		self._max = self._max.long()
		self.n = n
	
	
	def standardize(self, vals):
		return vals/self.n
	
	
	def unstandardize(self, vals):
		return (vals * self.n).long()
	
	
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


	def difference(self, x, y):
		return x.sub(y).bool().long()



class JointSpace(DimSpec):
	def __init__(self, *dims, shape=None, max=None, min=None, **kwargs):
		
		shape = (sum(len(dim) for dim in dims),)
		expanded_shape = sum(dim.expanded_len() for dim in dims)
		
		super().__init__(shape=shape, min=None, max=None, **kwargs)
		
		self._expanded_shape = expanded_shape
		self.dims = dims
		self._is_dense = any(1 for dim in dims if isinstance(dim, DenseDim))
	
	
	@property
	def expanded_shape(self):
		return self._expanded_shape
	
	
	def _dispatch(self, method, *vals, use_expanded=False, **base_kwargs):
		
		outs = []
		idx = 0
		B = None
		for dim in self.dims:
			D = dim.expanded_len() if use_expanded else len(dim)
			args = tuple(v.narrow(-1, idx, D) for v in vals)
			kwargs = base_kwargs.copy()
			
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
		return self._dispatch('sample', N=N, gen=gen, seed=seed)


	def difference(self, x, y):
		return self._dispatch('difference', x, y)


	def __getitem__(self, item):
		return self.dims[item]



class _DimSpec(Configurable, DimSpec):
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
class _HalfBoundDim(_DimSpec, HalfBoundDim):
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
class _BoundDim(_DimSpec, BoundDim):
	def __init__(self, A, epsilon=unspecified_argument, **kwargs):
		if epsilon is unspecified_argument:
			epsilon = A.pull('epsilon', 1e-10)
		super().__init__(A, epsilon=epsilon, **kwargs)



@fig.Component('space/unbound')
class _UnboundDim(_DimSpec, UnboundDim):
	pass



@fig.Component('space/periodic')
class _PeriodicDim(_DimSpec, PeriodicDim):
	def __init__(self, A, period=unspecified_argument, **kwargs):
		if period is unspecified_argument:
			period = A.pull('period', 1.)
			
		super().__init__(A, period=period, **kwargs)



@fig.Component('space/categorical')
class _CategoricalDim(_DimSpec, CategoricalDim):
	def __init__(self, A, n=unspecified_argument, **kwargs):
		if n is unspecified_argument:
			n = A.pull('n')
			
		super().__init__(A, n=n, **kwargs)



@fig.Component('space/joint')
class _JointSpace(_DimSpec, JointSpace):
	def __init__(self, A, dims=unspecified_argument, **kwargs):
		if dims is unspecified_argument:
			dims = A.pull('dims')
		
		super().__init__(A, _req_args=dims)
	




