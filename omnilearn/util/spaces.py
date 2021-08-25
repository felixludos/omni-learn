
from omnibelt import unspecified_argument
import omnifig as fig

import numpy as np
import torch
from torch.nn import functional as F

from .features import DeviceBase, Configurable

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
	
	
	def standardize(self, vals):
		raise NotImplementedError
	
	
	def unstandardize(self, vals):
		raise NotImplementedError
	
	

class DenseDim(DimSpec):
	
	def sample(self, N=None, gen=None, seed=None):
		if seed is not None:
			gen = torch.Generator()
			gen.manual_seed(seed)
		
		sqz = N is None
		if N is None:
			N = 1
		
		samples = torch.randn(N, *self.shape, generator=gen)
		return samples.squeeze(0) if sqz else samples
		


class HalfBoundDim(DenseDim):
	def __init__(self, bound=0, side='lower', sample_type='exp',
	             min=None, max=None, **kwargs):
		
		if min is None and max is None:
			assert bound is not None, 'No bound provided'
			assert side in {'lower', 'upper'}, 'Bound side not specified'
		
			if side == 'lower':
				min = bound
			else:
				max = bound
				
		assert max is None or min is None, f'Too many bounds specified: min={min}, max={max}'
		assert sample_type in {'exp', 'soft', 'chi', 'abs'}
		super().__init__(max=max, min=min, **kwargs)
		
		self._sample_type = sample_type
		
	
	def sample(self, N=None, gen=None, seed=None):
		
		samples = super().sample(N=N, gen=gen, seed=seed)
		
		if self._sample_type == 'soft':
			samples = F.softplus(samples)
		elif self._sample_type == 'chi':
			samples = samples.pow(2)
		elif self._sample_type == 'abs':
			samples = samples.abs()
		else:
			samples = samples.exp()
		
		if self.min is not None:
			samples = samples + self.min.unsqueeze(0)
		elif self.max is not None:
			samples = -samples + self.max.unsqueeze(0)
		
		return samples
	
	
	def standardize(self, vals):
		raise NotImplementedError
	
	
	def unstandardize(self, vals):
		raise NotImplementedError



class BoundDim(DenseDim):
	def __init__(self, min=0., max=1., **kwargs):
		super().__init__(min=min, max=max, **kwargs)
		assert self.min is not None, f'No lower bound provided'
		assert self.max is not None, f'No upper bound provided'
	
	
	def sample(self, N=None, gen=None, seed=None):
		if seed is not None:
			gen = torch.Generator()
			gen.manual_seed(seed)
		
		# kwargs = {} if gen is None else {'gen': gen}
		
		sqz = N is None
		if N is None:
			N = 1
		
		samples = self.unstandardize(torch.rand(N, *self.shape, generator=gen))
		return samples.squeeze(0) if sqz else samples


	def standardize(self, vals):
		return vals.sub(self.min.unsqueeze(0)).div(self.range.unsqueeze(0))
	
	
	def unstandardize(self, vals):
		return vals.mul(self.range.unsqueeze(0)).add(self.min.unsqueeze(0))



class UnboundDim(DenseDim):
	def __init__(self, min=None, max=None, **kwargs):
		super().__init__(min=None, max=None, **kwargs)
	
	
	def standardize(self, vals):
		raise NotImplementedError
	
	
	def unstandardize(self, vals):
		raise NotImplementedError



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
	
	
	def _dispatch(self, method, vals=None, use_expanded=False, **kwargs):
		
		outs = []
		idx = 0
		B = None
		for dim in self.dims:
			D = dim.expanded_len() if use_expanded else len(dim)
			args = kwargs.copy()
			if vals is not None:
				val = vals.narrow(-1, idx, D)
				args['vals'] = val
			
			out = getattr(dim, method)(**args)
			if B is None:
				B = out.size(0)
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
	             sample_type=unspecified_argument, **kwargs):
		
		if bound is unspecified_argument:
			bound = A.pull('bound', 0.)
		if side is unspecified_argument:
			side = A.pull('side', 'lower')
		if sample_type is unspecified_argument:
			sample_type = A.pull('sample-type', 'exp')
		
		super().__init__(A, bound=bound, side=side, sample_type=sample_type, **kwargs)



@fig.Component('space/bound')
class _BoundDim(_DimSpec, BoundDim):
	pass



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
	




