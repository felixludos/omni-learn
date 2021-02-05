import torch
from torch import nn

import omnifig as fig

# from .. import framework as fm
from ..op import framework as fm
from .. import util


@fig.AutoComponent('multigroup-cls')
class MultiGroupClassification(fm.FunctionBase):
	def __init__(self, group_sizes, group_weights=None):
		super().__init__(sum(group_sizes), 1)
		
		self.sizes = group_sizes
		
		self.criterion = nn.CrossEntropyLoss()
		
		if group_weights is not None:
			raise NotImplementedError
		
	def extra_repr(self) -> str:
		return 'groups = {}'.format(str(self.sizes))
		
	def get_sizes(self):
		return self.sizes
	
	def __len__(self):
		return len(self.sizes)
		
	def forward(self, pred, lbls):
		loss = 0
		for i, w in enumerate(self.split(pred)):
			y = lbls[:,i]
			loss += self.criterion(w, y)
		return loss
		
	def split(self, pred):
		return pred.split(self.sizes, dim=1)
	
	def _get_info(self, pred):
		return zip(*[s.max(dim=1) for s in self.split(pred)])

	def get_info(self, pred):
		return [torch.stack(info, dim=1) for info in self._get_info(pred)]

	def get_confidences(self, pred):
		return torch.stack(self._get_info(pred)[0], dim=1)
	
	def get_picks(self, pred):
		return torch.stack(self._get_info(pred)[1], dim=1)


class Feature_Match(nn.Module):

	def __init__(self, layers, criterion='mse', weights=None,
	             out_wt=None, out_criterion=None,
	             model=None, reduction='mean'):
		super().__init__()

		self._features = []

		def hook(m, input, out):
			self._features.append(out)

		L = 0
		for layer in layers:
			L += 1
			layer.register_forward_hook(hook)
		self.num = L

		if weights is not None:
			assert L == len(weights), '{} != {}'.format(len(layers), len(weights))
			self.weights = weights
			# self.register_buffer('weights', weights)
		else:
			self.weights = None
		self.criterion = util.get_loss_type(criterion)
		assert reduction in {'mean', 'sum', 'none'}
		self.reduction = reduction

		self.out_wt = out_wt
		self.out_criterion = util.get_loss_type(out_criterion)
		if self.out_wt is not None and self.out_wt > 0:
			self.num += 1

		self._model = [model] # dont let pytorch treat this as a submodule

	def __len__(self):
		return self.num

	def clear(self):
		self._features.clear()

	def forward(self, p, q, model=None):

		if model is None:
			model = self._model[0]

		self.clear()
		po = model(p)
		pfs = self._features.copy()

		self.clear()
		qo = model(q)
		qfs = self._features.copy()

		self.clear()

		if self.weights is None:
			losses = [self.criterion(pf, qf) for pf, qf in zip(pfs, qfs)]
		else:
			losses = [w*self.criterion(pf, qf) for w, pf, qf in zip(self.weights, pfs, qfs)]

		if self.out_wt is not None and self.out_wt > 0:
			criterion = self.criterion if self.out_criterion is None else self.out_criterion
			out_loss = self.out_wt * criterion(po, qo)

			losses.append(out_loss)

		if self.reduction == 'none':
			return losses

		loss = sum(losses)

		if self.reduction == 'mean':
			return loss / len(losses)

		return loss



