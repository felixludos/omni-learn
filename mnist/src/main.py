
import sys, os
import time

import torch
from torch import nn
import omnifig as fig

import omnilearn as learn
from omnilearn import util
from omnilearn import models
from omnidata.framework import machine, hparam, inherit_hparams, spaces, get_builder
from omnidata import framework as fm
# from omnidata.nn import MLP
from omnilearn.novo import trainers
from omnilearn.novo import MLP


@fig.Script('test')
def _test_script(A):

	print(A.pull('a'))
	print(A.pull('x', 10))
	print(A.pulls('y', 'y1', 'y2', default=100))
	print(A.pulls('b2', 'b'))

	print('done')


class Supervised_Model(trainers.SimplePytorchModel):
	def __init__(self, *args, target_space=None, criterion=None, **kwargs):
		if target_space is None:
			target_space = self.dout
		if criterion is None:
			criterion = get_builder('loss').build(target_space)
		super().__init__(*args, **kwargs)
		self.criterion = criterion
	

	optimizer = machine(builder='optimizer')

	
	# class Statistics(fm.SimplePytorchModel.Statistics):
	#
	# 	def _compute_simple_stats(self, info, **kwargs):
	# 		return Ord
	#
	# 	def mete(self, info, **kwargs):
	#
	# 		pass
	
	
	def _compute_loss(self, info, **kwargs):
		info['pred'] = self(info['observation'])
		
		if 'loss' not in info:
			info['loss'] = 0.
		info['loss'] += self.criterion(info['pred'], info['target'])
		
		return info
	
	
	# _input_key = 'observation'
	# _pred_key = 'pred'
	# _target_key = 'target'
	#
	#
	# def _step(self, info, take_step=True, **kwargs):
	# 	info[self._pred_key] = self(info[self._input_key])
	#
	# 	if 'loss' not in info:
	# 		info['loss'] = 0.
	# 	info['loss'] += self.criterion(info[self._pred_key], info[self._target_key])
	#
	# 	if take_step and self.training:
	# 		self.optimizer.zero_grad()
	# 		info['loss'].backward()
	# 		self.optimizer.step()
	#
	# 	return info



# @inherit_hparams(#'lr', 'weight_decay',
#                  'nonlin', 'norm', 'dropout', 'bias', 'out_nonlin', 'out_norm', 'out_bias')
@inherit_hparams('optimizer', 'nonlin', 'norm', 'dropout', 'bias', 'out_nonlin', 'out_norm', 'out_bias')
class Supervised_Model(Supervised_Model, MLP):
	width = hparam(64, space=[64, 128, 256, 512, 1024])
	depth = hparam(1, space=[0, 1, 2, 3, 4, 6, 8])

	
	@hparam(hidden=True)
	def hidden(self):
		return [self.width] * self.depth
	
	
	# class Statistics(Supervised_Model.Statistics):
	
	# def _extract_stats(self, info):
	# 	stats = super()._extract_stats(info)
	#
	# 	if isinstance(self.dout, spaces.Categorical):
	# 		confs, picks = info['pred'].max(-1)
	# 		stats['accuracy'] = (picks == info['target']).float().mean()
	# 		stats['confidence'] = confs#.mean()
	# 	else:
	# 		raise NotImplementedError
	#
	# 	return stats



# @fig.Component('simple')
class Old_Simple_Model(learn.Model):
	def __init__(self, info):

		net = info.pull('net')
		criterion = info.pull('criterion', 'cross-entropy')

		super().__init__(info, din=net.din, dout=net.dout)

		self.net = net
		self.criterion = models.get_loss_type(criterion)

		self.register_stats('accuracy', 'confidence')

	def forward(self, x):
		return self.net(x)

	def _visualize(self, info, records):
		
		x, y, pred  = info.x, info.y, info.pred
		N = 24
		
		guess = pred[:N].max(-1)[1]
		
		fg, ax = util.plot_imgs(x[:N], titles=guess[:N].tolist())
		
		records.log('figure', 'samples', fg)

	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		# compute loss
		x, y = batch

		out.x, out.y = x, y

		pred = self(x)
		out.pred = pred

		conf, pick = pred.max(-1)
		confidence = conf.detach()
		correct = pick.sub(y).eq(0).float().detach()
		self.mete('confidence', confidence.mean())
		self.mete('accuracy', correct.mean())

		loss = self.criterion(pred, y)
		out.loss = loss

		if self.train_me():
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out






if __name__ == '__main__':
	fig.entry()


