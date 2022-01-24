
import numpy as np
from torch.nn import functional as F

from ..util import Distribution



def log_likelihood(original, reconstruction, batch_mean=True):
	if isinstance(reconstruction, Distribution):
		return reconstruction.log_prob(original).view(original.size(0),-1).sum(-1).mean()
	if len(reconstruction.shape) > 2:
		return F.binary_cross_entropy(reconstruction, original, reduction='none')\
			.view(original.size(0),-1).sum(-1).mean()
	return F.mse_loss(reconstruction, original).view(original.size(0),-1).sum(-1).mean()


def elbo(original, reconstruction, kl, batch_mean=True):
	ll = log_likelihood(original, reconstruction, batch_mean=batch_mean)
	return ll - kl


def bits_per_dim(original, reconstruction, batch_mean=True):
	ll = log_likelihood(original, reconstruction, batch_mean=batch_mean)
	return ll / np.log(2) / int(np.prod(original.shape))