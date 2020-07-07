
import sys, os
import omnifig as fig

import foundation as fd
from foundation import util

@fig.Component('simple')
class Simple_Model(fd.Visualizable, fd.Schedulable, fd.Trainable_Model):
	def __init__(self, info):

		net = info.pull('net')
		criterion = info.pull('criterion', 'cross-entropy')

		super().__init__(net.din, net.dout)

		self.net = net
		self.criterion = util.get_loss_type(criterion)

		self.stats.new('accuracy', 'confidence')

	def forward(self, x):
		return self.net(x)

	def _visualize(self, info, logger):

		conf, pick = info.pred.max(-1)

		confidence = conf.detach()
		correct = pick.sub(info.y).eq(0).float().detach()

		self.stats.update('confidence', confidence.mean())
		self.stats.update('accuracy', correct.mean())


	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		# compute loss
		x, y = batch

		out.x, out.y = x, y

		pred = self(x)
		out.pred = pred

		loss = self.criterion(pred, y)
		out.loss = loss

		if self.train_me():
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out

if __name__ == '__main__':
	fig.entry()


