from omnibelt import agnosticmethod
import torch
from torch import nn

from omnidata import hparam, module, inherit_hparams, with_hparams, Parameterized, \
	get_builder, Prepared, TrainableModel

from .optim import PytorchOptimizer


class PytorchModel(TrainableModel, nn.Module):
	@agnosticmethod
	def step(self, info, **kwargs):
		if not self.training:
			self.train()
		return super().step(info, **kwargs)


	@agnosticmethod
	def eval_step(self, info, **kwargs):
		if self.training:
			self.eval()
		with torch.no_grad():
			return super().eval_step(info, **kwargs)



class SimplePytorchModel(PytorchModel):
	_loss_key = 'loss'

	optimizer = module(type=PytorchOptimizer, builder='optimizer')


	def _prepare(self, source=None, **kwargs):
		out = super()._prepare(source=source, **kwargs)
		self.optimizer.prepare(self.parameters())
		return out


	def _compute_loss(self, info):
		return info


	def _step(self, info):
		self._compute_loss(info)

		if self.training:
			loss = info[self._loss_key]

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		return info

 


class SimpleTrainer(Trainer):

	pass









