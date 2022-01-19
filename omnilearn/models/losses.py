
import numpy as np

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as distrib
from torch.nn.modules.loss import _Loss

prt = get_printer(__file__)

try:
	import pytorch_msssim
except ImportError:
	prt.warning('Pytorch MS-SSIM not found: install with "pip install pytorch-msssim"')


from ..op import framework as fm
from .. import util



def log_likelihood(reconstruction, original, reduction='batch'):
	if isinstance(reconstruction, util.Distribution):
		ll = reconstruction.log_prob(original)
	elif len(reconstruction.shape) > 2:
		ll = F.binary_cross_entropy(reconstruction, original, reduction='none')
	else:
		ll = F.mse_loss(reconstruction, original)

	if reduction == 'none':
		return ll
	ll = ll.view(original.size(0),-1).sum(-1)
	if reduction == 'sum':
		return ll.sum()
	return ll.mean()



def elbo(reconstruction, original, kl, reduction='batch'):
	ll = log_likelihood(reconstruction, original, reduction=reduction)
	return ll - kl



def bits_per_dim(reconstruction, original, reduction='batch'):
	ll = log_likelihood(reconstruction, original, reduction=reduction)
	return ll / np.log(2) / int(np.prod(original.shape))



class Metric:
	def distance(self, a, b):
		raise NotImplementedError



class EncodedMetric(fm.Function, Metric, fm.Encodable):
	def __init__(self, A, criterion=unspecified_argument, **kwargs):
		if criterion is unspecified_argument:
			criterion = A.pull('criterion', 'mse')
		super().__init__(A, **kwargs)
		self.criterion = models.get_loss_type(criterion, reduction='none')


	def distance(self, a, b):
		return self.criterion(self.encode(a), self.encode(b))



class Loss(fm.FunctionBase):
	def __init__(self, reduction='mean', **kwargs):
		super().__init__(**kwargs)
		self.reduction = reduction


	def set_reduction(self, reduction):
		self.reduction = reduction


	def extra_repr(self) -> str:
		return f'reduction={self.reduction}'


	def forward(self, input, *args, **kwargs):
		loss = self._forward(input, *args, **kwargs)

		if self._reduction == 'batch-mean':
			loss = loss.contiguous().view(input.size(0), -1).sum(-1).mean()
		if self._reduction == 'sample-mean':
			loss = loss.contiguous().view(input.size(0), -1).mean(-1)
		if self._reduction == 'sample-sum':
			loss = loss.contiguous().view(input.size(0), -1).sum(-1)
		if self._reduction == 'sum':
			loss = loss.sum()
		if self._reduction == 'mean':
			loss = loss.mean()
		return loss

	def _forward(self, input, *args, **kwargs):
		raise NotImplementedError



@fig.AutoModifier('loss/invert')
class Inverted(Loss):
	def forward(self, *args, **kwargs):
		return 1 / super().forward(*args, **kwargs)



@fig.AutoModifier('loss/negative')
class Negative(Loss):
	def forward(self, *args, **kwargs):
		return - super().forward(*args, **kwargs)



@fig.Component('ms-ssim')
class MSSSIM(Loss, pytorch_msssim.MS_SSIM):
	def __init__(self, size_average=False, _req_kwargs={}, **kwargs):
		super().__init__(_req_kwargs={'size_average':size_average, **_req_kwargs}, **kwargs)


	def _forward(self, img1, img2):
		return super(Loss, self).forward(img1, img2)



@fig.Component('psnr')
class PSNR(Loss):
	@staticmethod
	def _forward(img1, img2):
		mse = img1.sub(img2).pow(2).view(img1.size(0), -1).mean(-1)
		return 20 * torch.log10(255.0**(img1.dtype == torch.uint8) / mse.sqrt())



@fig.Component('frechet-distance')
class FrechetDistance(Loss):
	def _forward(self, p, q):
		return util.frechet_distance(p, q)



@fig.Component('bits-per-dim')
class BitsPerDim(Loss):
	def _forward(self, reconstruction, original, **kwargs):
		return bits_per_dim(reconstruction, original, reduction='none')



@fig.Component('elbo')
class ELBO(Loss):
	def _forward(self, reconstruction, original, **kwargs):
		return elbo(reconstruction, original, reduction='none')



class PytorchLoss(Loss, _Loss):
	def __init__(self, reduction='mean', **kwargs):
		# div_batch = False
		# sample_batch = False
		# if 'batch' in reduction:
		# 	div_batch = True
		# 	reduction = 'sum'
		# if 'sample' in reduction:
		# 	sample_batch = True
		# 	reduction = 'none'
		# 	div_batch = False
		super().__init__(reduction='none', **kwargs)
		self._reduction = reduction
		# self._div_batch = div_batch
		# self._sample_batch = sample_batch


	def set_reduction(self, reduction):
		self._reduction = reduction


	def extra_repr(self) -> str:
		return f'reduction={self._reduction}'
		# reduction = 'batch-mean' if self._div_batch else self.reduction
		# reduction = 'sample-mean' if self._sample_batch else reduction
		# return f'reduction={reduction}'


	def _forward(self, input, *args, **kwargs):
		if isinstance(input, distrib.Distribution):
			input = input.rsample()
		return super(Loss, self).forward(input, *args, **kwargs)




@fig.AutoComponent('distrib-nll')
class DistributionNLLLoss(Loss):
	def __init__(self, mn_lim=None, mx_lim=None, **kwargs):
		super().__init__(**kwargs)
		self._mn_lim = mn_lim
		self._mx_lim = mx_lim
	
	def forward(self, dis, target):
		
		if self._mn_lim is not None or self._mx_lim is not None:
			target = target.clamp(min=self._mn_lim, max=self._mx_lim)
		
		ll = dis.log_prob(target)
		
		if self.reduction == 'mean':
			return -ll.mean()
		elif self.reduction == 'sum':
			loss = -ll.sum()
			return loss.div(ll.size(0)) if self._div_batch else loss
		return -ll



class NormMetric(Metric, Loss):
	def distance(self, a, b):
		dists = self._forward(a-b).view(a.size(0))
		return dists



class Lp_Norm(NormMetric):
	def __init__(self, p=2, dim=None, reduction='mean', **kwargs):
		super().__init__(reduction=reduction, **kwargs)
		self.p = p
		self.dim = dim


	def extra_repr(self):
		return f'p={self.p}'


	def _forward(self, input, *args, **kwargs):
		return input.norm(p=self.p, dim=self.dim)



class MSELoss(PytorchLoss, nn.MSELoss):
	pass



class RMSELoss(MSELoss):
	def forward(self, *args, **kwargs):
		loss = super().forward(*args, **kwargs)
		return loss.sqrt()



class L1Loss(PytorchLoss, nn.L1Loss):
	pass



class SmoothL1Loss(PytorchLoss, nn.SmoothL1Loss):
	pass



class NLLLoss(PytorchLoss, nn.NLLLoss):
	pass



class CrossEntropyLoss(PytorchLoss, nn.CrossEntropyLoss):
	pass



class KLDivLoss(PytorchLoss, nn.KLDivLoss):
	pass



class BCELoss(PytorchLoss, nn.BCELoss):
	pass



class BCEWithLogitsLoss(PytorchLoss, nn.BCEWithLogitsLoss):
	pass


class CosineSimilarity(PytorchLoss, nn.CosineSimilarity):
	pass



@fig.AutoComponent('criterion') # TODO: legacy
@fig.AutoComponent('loss')
def get_loss_type(ident, **kwargs):

	if not isinstance(ident, str):
		return ident

	if ident == 'mse':
		return MSELoss(**kwargs)
	elif ident == 'distrib-nll':
		return DistributionNLLLoss(**kwargs)
	elif ident == 'rmse':
		return RMSELoss(**kwargs)
	elif ident == 'l1':
		return L1Loss(**kwargs)
	elif ident == 'lp':
		return Lp_Norm(**kwargs)
	elif ident == 'l2':
		return Lp_Norm(p=2, **kwargs)
	elif ident == 'huber':
		return SmoothL1Loss(**kwargs)
	elif ident == 'nll':
		print('WARNING: should probably use cross-entropy')
		return NLLLoss(**kwargs)
	elif ident == 'cross-entropy':
		return CrossEntropyLoss(**kwargs)
	elif ident == 'kl-div':
		return KLDivLoss(**kwargs)
	elif ident == 'bce':
		#print('WARNING: should probably use bce-log')
		return BCELoss(**kwargs)
	elif ident == 'bce-log':
		return BCEWithLogitsLoss(**kwargs)
	elif ident == 'frechet-distance':
		return FrechetDistance(**kwargs)
	elif ident == 'ms-ssim':
		return MSSSIM(**kwargs)
	elif ident == 'cosine-similarity':
		return CosineSimilarity(**kwargs)
	else:
		assert False, "Unknown loss type: " + ident



@fig.AutoComponent('viz-criterion')
class Viz_Criterion(nn.Module):
	def __init__(self, criterion, arg_names=[], kwarg_names=[],
				 allow_grads=False):
		super().__init__()
		
		self.criterion = get_loss_type(criterion)
		self.arg_names = arg_names
		self.kwarg_names = kwarg_names
		self.allow_grads = allow_grads
	
	def forward(self, out):
		args = [out[key] for key in self.arg_names]
		kwargs = {key: out[key] for key in self.kwarg_names}
		
		if self.allow_grads:
			return self.criterion(*args, **kwargs)
		
		with torch.no_grad():
			return self.criterion(*args, **kwargs)








