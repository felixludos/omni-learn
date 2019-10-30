
import sys, os, time, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import configargparse

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('Agg')

import foundation as fd
from foundation import models
from foundation import util
from foundation import train
from foundation import data
from foundation.models import unsup


class Feature_Discriminator(fd.Trainable_Model):
	def __init__(self, input_dim,
	             nonlin='prelu', hidden_dims=[64], flatten=True,
	             gp_weight=None, criterion='mse'):

		super().__init__(input_dim, 1)

		assert len(hidden_dims), 'must have hidden dims for feature match'


		self._feature_list = []

		def hook(m, input, out):
			self._feature_list.append(out)

		self.net = models.make_MLP(input_dim=input_dim, output_dim=1,
		                           hidden_dims=hidden_dims,
		                           nonlin=nonlin)

		for layer in self.net[:-1]:
			if isinstance(layer, nn.Linear):
				layer.register_forward_hook(hook)

		self.criterion = util.get_loss_type(criterion)
		self.flatten = nn.Flatten() if flatten else None
		self.gp_weight = gp_weight

		self.stats.new('wasserstein')
		if self.gp_weight is not None and self.gp_weight > 0:
			self.stats.new('reg')

	def distance(self, x, y):
		self._feature_list.clear()
		self(x)
		xf = self._feature_list.copy()

		self._feature_list.clear()
		self(y)
		yf = self._feature_list.copy()

		loss = 0
		for xt, yt in zip(xf, yf):
			loss += self.criterion(xt, yt)

		self._feature_list.clear()

		return loss

	def encode_output(self, y):
		return y

	def forward(self, x):
		if self.flatten is not None:
			x = self.flatten(x)
		return self.net(x)

	def _visualize(self, info, logger):
		pass

	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		real, fake = batch.real.detach(), batch.fake.detach()

		loss = 0
		if self.gp_weight is not None and self.gp_weight > 0:
			gp_loss = unsup.grad_penalty(self, real, fake)
			out.gp_loss = gp_loss
			self.stats.update('reg', gp_loss.detach())
			loss += self.gp_weight * gp_loss

		vreal = self(real)
		vfake = self(fake)

		wasserstein = vreal.mean() - vfake.mean()
		out.wasserstein = wasserstein
		self.stats.update('wasserstein', wasserstein.detach())
		loss -= wasserstein

		out.loss = loss

		if self.train_me():
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		self._feature_list.clear()

		return out

class Class_Discriminator(Feature_Discriminator):
	def __init__(self, input_dim, embedding_dim, num_classes, **kwargs):

		full_input_dim = input_dim + embedding_dim

		super().__init__(input_dim=full_input_dim, **kwargs)

		self.table = nn.Embedding(num_classes, embedding_dim)

		self.stats.new('accuracy', 'confidence')

	def compute_accuracy(self, pred, y):

		with torch.no_grad():

			pred = pred.unsqueeze(0)
			pts = self.table.weight.unsqueeze(1)

			dists = (pred - pts).pow(2).sum(-1)

			inv_conf, pick = dists.min(0)

			conf = 1./(inv_conf+1e-6)
			correct = pick.sub(y).eq(0).float()

			self.stats.update('accuracy', correct.mean())
			self.stats.update('confidence', conf.mean())

		return correct, conf

	def _visualize(self, info, logger):
		if False and self.table.weight.size(1) == 2:
			fig = plt.figure(figsize=(3, 3))
			cls = self.table.weight.data.cpu().numpy()
			pts = info.pred.detach().cpu().numpy()
			plt.gca().set_aspect('equal')
			plt.scatter(*pts.T, marker='.', s=6, edgecolors='none', alpha=.7)
			plt.scatter(*cls.T)
			for i in range(cls.shape[0]):
				plt.annotate(str(i), cls[i])
			logger.add('figure', 'clusters', fig, close=True)


		self.compute_accuracy(info.pred, info.y)

	def encode_output(self, y):
		return self.table(y)

class Supervised_LL(fd.Encodable, fd.Visualizable, fd.Trainable_Model):

	def __init__(self, func, encoder, discriminator, disc_steps=1):
		super().__init__(func.din, func.dout)

		self.func = func
		self.enc = encoder
		self.disc = discriminator

		if isinstance(self.disc, fd.Recordable):
			self.stats.shallow_join(self.disc.stats)

		self.disc_steps = disc_steps
		self.step_counter = 0

	def encode(self, x):
		return self.enc(x)

	def forward(self, x):
		return self.func(x)

	def _visualize(self, info, logger):
		self.disc._visualize(info, logger)
		pass

	def _step(self, batch, out=None):
		self.step_counter += 1
		if out is None:
			out = util.TensorDict()

		x, y = batch

		out.x = x
		out.y = y

		pred = self(x)
		out.pred = pred

		q = self.encode(x)
		out.q = q

		fake = torch.cat([pred, q], -1)
		real = torch.cat([self.disc.encode_output(y), q], -1)

		out.real, out.fake = real, fake

		self.disc._step(out, out=out)

		disc_loss = out.loss
		del out.loss

		if self.train_me():
			self.optim.disc.zero_grad()
			self.optim.enc.zero_grad()
			disc_loss.backward()
			self.optim.disc.step()
			self.optim.enc.step()

		out.disc_loss = disc_loss

		loss = self.disc.distance(fake, real)

		out.loss = loss

		if self.train_me() and (self.disc_steps is None or self.step_counter % self.disc_steps == 0):
			self.optim.func.zero_grad()
			loss.backward()
			self.optim.func.step()

		return out





def get_options(parser=None):
	if parser is None:
		parser = train.get_parser()

	parser.add_argument('--disc-steps', type=int, default=1)
	parser.add_argument('--disc-gp', type=float, default=None)


	parser.add_argument('--emb-dim', type=int, default=None)
	# parser.add_argument('--latent-dim', type=int, default=16)


	parser.add_argument('--criterion', type=str, default='mse')


	parser.add_argument('--disc-hidden', type=int, nargs='+', default=[])
	parser.add_argument('--enc-hidden', type=int, nargs='+', default=[])
	parser.add_argument('--func-hidden', type=int, nargs='+', default=[])


	return parser

def get_data(args):
	return train.load_data(args=args)

def get_model(args):

	args.din_flat = int(np.product(args.din))

	encoder = nn.Sequential(nn.Flatten(), models.make_MLP(args.din_flat, args.latent_dim,
	                                                      hidden_dims=args.enc_hidden, nonlin=args.nonlin))

	net = nn.Sequential(nn.Flatten(), models.make_MLP(args.din_flat, args.emb_dim,
	                                                      hidden_dims=args.func_hidden, nonlin=args.nonlin))
	net.din, net.dout = args.din_flat, args.emb_dim

	discriminator = Class_Discriminator(input_dim=args.latent_dim, embedding_dim=args.emb_dim, num_classes=args.dout,
	                                    hidden_dims=args.disc_hidden, nonlin=args.nonlin,
	                                    gp_weight=args.disc_gp, criterion=args.criterion,
	                                    flatten=True)

	model = Supervised_LL(net, encoder, discriminator, disc_steps=args.disc_steps)

	model.set_optim(util.Complex_Optimizer(
		enc=util.get_optimizer(args.optim_type, encoder.parameters(),
		                       lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
		func=util.get_optimizer(args.optim_type, net.parameters(),
		                        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
		disc=util.get_optimizer(args.optim_type, discriminator.parameters(),
		                        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
	))

	return model


if __name__ == '__main__':

	argv = None

	argv = ['', '--config', 'config/test.yaml', '']

	try:
		train.run_full(get_options, get_data, get_model, argv=argv)

	except KeyboardInterrupt:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()

	except:
		extype, value, tb = sys.exc_info()
		traceback.print_exc()
		ipdb.post_mortem(tb)




