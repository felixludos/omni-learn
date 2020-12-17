
import sys, os

import torch
import omnifig as fig

import foundation as fd
from foundation import util

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

	def _evaluate(self, info):
		
		results = {}
		
		logger = info.logger
		
		A = info._A
		device = A.pull('device', 'cpu')
		
		loader = iter(info.testloader)
		total = 0
		
		batch = next(loader)
		batch = util.to(batch, device)
		total += batch.size(0)
		
		with torch.no_grad():
			out = self.test(batch)
		
		if isinstance(self, fd.Visualizable):
			self.visualize(out, logger)
		
		results['out'] = out
		
		for batch in loader:  # complete loader for stats
			batch = util.to(batch, device)
			total += batch.size(0)
			with torch.no_grad():
				self.test(batch)
		
		results['stats'] = self.stats.export()
		display = self.stats.avgs()  # if smooths else stats.avgs()
		for k, v in display.items():
			logger.add('scalar', k, v)
		results['stats_num'] = total
		
		return results

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


