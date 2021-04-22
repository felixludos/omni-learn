
import sys, os
import time

import torch
import omnifig as fig

import omnilearn as fd
from omnilearn import util

@fig.Component('simple')
class Simple_Model(fd.Model):
	def __init__(self, info):

		net = info.pull('net')
		criterion = info.pull('criterion', 'cross-entropy')

		super().__init__(info, din=net.din, dout=net.dout)

		self.net = net
		self.criterion = util.get_loss_type(criterion)

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


