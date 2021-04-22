import sys, os, time
import random
import numpy as np
import torch
import torch.multiprocessing as mp

from omnibelt import unspecified_argument
import omnifig as fig

from .collectors import Dataset

from .. import util


class SamplerBase:
	def __init__(self):
		self.units = None

	def _set_units(self):
		sz = self.factors_num_values.tolist()
		sz.append(1)
		self.units = torch.from_numpy(np.cumprod(sz[::-1])[-2::-1].copy()).long()

	def __len__(self):
		if self.units is None:
			self._set_units()
		return len(self.units)
	
	def sample_labels(self, batch, random_state=None):
		# if random_state is None:
		# 	random_state = torch
		sizes = self.factors_num_values
		return torch.rand(batch, len(sizes)).mul(sizes).long()
	
	def labels_to_inds(self, labels):
		if isinstance(labels, np.ndarray):
			labels = torch.from_numpy(labels)
		if self.units is None:
			self._set_units()
		return labels.matmul(self.units.unsqueeze(-1)).squeeze()
	
	@property
	def num_factors(self):
		return len(self.factors_num_values)
	
	@property
	def factors_num_values(self):
		raise NotImplementedError
		
	@property
	def observation_shape(self):
		raise NotImplementedError
	
	def inds_to_samples(self, inds):
		return self[inds]


class InterventionSamplerBase(SamplerBase):
	def __init__(self, dataset, include_labels=False):
		super().__init__()
		self.dataset = dataset
		self.include_labels = include_labels
	
	@property
	def factors_num_values(self):
		sizes = self.dataset.get_factor_sizes()
		if isinstance(sizes, list):
			sizes = torch.tensor(sizes)
		return sizes
	
	@property
	def observation_shape(self):
		return self.dataset.din
	
	def inds_to_samples(self, inds):
		imgs, lbls = self.dataset[inds]
		if self.include_labels:
			return imgs, lbls
		return imgs
	
	def full_intervention(self, idx=None, B=None, vals=None):
		'''
		Intervenes on all but label-dim `idx`
		
		:param idx: index of the label-dim that should be resampled
		:param B: `B` samples (default is size of latent-dim `idx`)
		:param vals: label vec to use (vals[idx] is ignored)
		:return: `B` samples of `vals` except where label-dim `idx` is resampled
		'''
		if idx is None:
			idx = random.randint(0, self.num_factors-1)
		if B is None:
			B = self.factors_num_values[idx]
		if vals is None:
			vals = self.sample_labels(1)
		sample = vals.view(1, self.num_factors).expand(B, self.num_factors).clone()
		sample[:, idx] = torch.arange(B)
		inds = self.labels_to_inds(sample)
		return self.inds_to_samples(inds)
	
	def intervention(self, idx=None, B=128, val=None):
		'''
		Intervene on label-dim `idx`
		
		:param idx: index of the label-dim that should be fixed to `val`
		:param B: number of samples to return
		:param val: value of label-dim `idx` that should be set
		:return: `B` random samples except where label-dim `idx` is set to `val`
		'''
		sizes = self.factors_num_values
		sample = self.sample_labels(B)
		if idx is None:
			idx = random.randint(0, self.num_factors-1)
		if val is None:
			val = torch.randint(sizes[idx], size=())
		sample[:, idx] = val
		inds = self.labels_to_inds(sample)
		return self.inds_to_samples(inds)


@fig.Component('intervention-sampler')
class InterventionSampler(util.Configurable, InterventionSamplerBase):
	def __init__(self, A, dataset=None, sizes=unspecified_argument, include_labels=None, **kwargs):
		
		if dataset is None:
			dataset = A.pull('dataset', ref=True)
		
		if sizes is unspecified_argument:
			sizes = A.pull('sizes', None)
		
		if include_labels is None:
			include_labels = A.pull('include-labels', False)
			
		super().__init__(A, dataset=dataset, sizes=sizes, include_labels=include_labels,
		                 _req_kwargs={'dataset':dataset, 'include_labels':include_labels}, **kwargs)


@fig.AutoModifier('joint-sampler')
class JointFactorSampler(Dataset, util.Configurable, SamplerBase):
	'''
	for datasets of the type: (observation, factors)
	where the observation is usually an image, and the factors are the latent factors of variation that (ideally)
	include all causal variables in the SCM of the underlying process that produced the observations.
	'''
	def __init__(self, A, to_numpy=None, **kwargs):
		
		if to_numpy:
			to_numpy = A.pull('to_numpy', False)
		
		super().__init__(A, **kwargs)
		
		self.to_numpy = to_numpy
	
	@property
	def factors_num_values(self):
		return torch.tensor(self.get_factor_sizes())
	
	@property
	def observation_shape(self):
		return self.din
	
	def inds_to_samples(self, inds):
		out = self[inds]
		if isinstance(out, tuple):
			return out[0]
		return out
	
	def sample_factors(self, num, random_state, to_numpy=None):
		"""Sample a batch of factors Y."""
		if to_numpy is None:
			to_numpy = self.to_numpy
		factors = self.sample_labels(num, random_state=random_state)
		if to_numpy:
			return factors.numpy()
		return factors
	
	def sample_observations_from_factors(self, factors, random_state, to_numpy=None):
		"""Sample a batch of observations X given a batch of factors Y."""
		if to_numpy is None:
			to_numpy = self.to_numpy
		observations = self.inds_to_samples(self.labels_to_inds(factors))
		if to_numpy:
			observations = observations.numpy()
		return observations
	
	def sample(self, num, random_state, to_numpy=None):
		"""Sample a batch of factors Y and observations X."""
		if to_numpy is None:
			to_numpy = self.to_numpy
		factors = self.sample_factors(num, random_state, to_numpy=False)
		observations = self.sample_observations_from_factors(factors, random_state)
		if to_numpy:
			factors = factors.numpy()
			observations = observations.numpy()
		return factors, observations
	
	def sample_observations(self, num, random_state, to_numpy=None):
		"""Sample a batch of observations X."""
		return self.sample(num, random_state, to_numpy=to_numpy)[1]
		





