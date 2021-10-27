
import numpy as np

from omnibelt import unspecified_argument
import omnifig as fig

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as distrib
from torch.nn.modules.loss import _Loss

from .. import util



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




class Loss(_Loss):
	def __init__(self, reduction='mean', div_batch=False, **kwargs):
		if reduction == 'batch-mean':
			div_batch = True
			reduction = 'sum'
		super().__init__(reduction=reduction, **kwargs)
		self._div_batch = div_batch
	
	def extra_repr(self) -> str:
		reduction = 'batch-mean' if self._div_batch else self.reduction
		return f'reduction={reduction}'

	def _forward(self, input, *args, **kwargs):
		return super().forward(input, *args, **kwargs)

	def forward(self, input, *args, **kwargs):
		if isinstance(input, distrib.Distribution):
			input = input.rsample()
		B = input.size(0) if self._div_batch else None
		loss = self._forward(input, *args, **kwargs)
		if B is not None:
			loss = loss / B
		return loss



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



@fig.Component('frechet-distance')
class FrechetDistance(Loss):
	def _forward(self, p, q):
		return util.frechet_distance(p, q)



class NormMetric(Metric):
	def distance(self, a, b):
		return self(a-b)



class Lp_Norm(NormMetric, Loss):
	def __init__(self, p=2, dim=None, reduction='mean', **kwargs):
		super().__init__(reduction=reduction, **kwargs)
		self.p = p
		self.dim = dim



	def extra_repr(self):
		return 'p={}'.format(self.p)


	def _forward(self, input, *args, **kwargs):
		return input.norm(p=self.p, dim=self.dim)



class MSELoss(Loss, nn.MSELoss):
	pass



class RMSELoss(MSELoss):
	def forward(self, *args, **kwargs):
		loss = super().forward(*args, **kwargs)
		return loss.sqrt()



class L1Loss(Loss, nn.L1Loss):
	pass



class SmoothL1Loss(Loss, nn.SmoothL1Loss):
	pass



class NLLLoss(Loss, nn.NLLLoss):
	pass



class CrossEntropyLoss(Loss, nn.CrossEntropyLoss):
	pass



class KLDivLoss(Loss, nn.KLDivLoss):
	pass



class BCELoss(Loss, nn.BCELoss):
	pass



class BCEWithLogitsLoss(Loss, nn.BCEWithLogitsLoss):
	pass


class CosineSimilarity(Loss, nn.CosineSimilarity):
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


