

import sys, os, time
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import gym
import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg') #('Qt5Agg')
import foundation as fd
from foundation import models
from foundation import util
from foundation import train
from foundation import data


class MLP_Decoder(fd.Decodable, fd.Model):

	def __init__(self, latent_dim, out_shape, hidden_dims=[], nonlin='prelu', output_nonlin=None):
		super().__init__(latent_dim, out_shape)

		self.out_shape = out_shape
		self.out_dim = np.product(out_shape)

		self.net = models.make_MLP(latent_dim, self.out_dim, hidden_dims=hidden_dims,
								   output_nonlin=output_nonlin, nonlin=nonlin)

	def forward(self, q):
		return self.decode(q)

	def decode(self, q):
		B = q.size(0)
		out = self.net(q)
		return out.view(B, *self.out_shape)

class DirectDecoder(fd.Generative, fd.Decodable, fd.Schedulable, fd.Vizualizable, fd.Model):

	def __init__(self, decoder, latent_dim, vocab_size, beta=1.,
				 zero_embedding=False,):
		super().__init__(decoder.din, decoder.dout)

		self.latent_dim = latent_dim
		self.beta = beta

		self.table = nn.Embedding(vocab_size, latent_dim)
		if zero_embedding:
			self.table.weight.data.mul_(0)

		self.dec = decoder
		# self.dec = models.Decoder(out_shape, latent_dim=latent_dim, **dec_kwargs)

		self.criterion = nn.BCELoss()

		self.stats.new('reg')

	def _visualize(self, info, logger):

		if self._viz_counter % 5 == 0:

			logger.add('histogram', 'latent-norm', info.latent.norm(p=2, dim=-1))

			B, C, H, W = info['original'].shape
			N = min(B, 8)

			viz_x, viz_rec = info['original'][:N], info['reconstruction'][:N]

			recs = torch.cat([viz_x, viz_rec],0)
			logger.add('images', 'rec', recs)

			# show some generation examples
			try:
				# gen = self.generate(16)
				q_gen = self.sample_prior(2*N)
				logger.add('histogram', 'gen-norm', q_gen.norm(p=2, dim=-1))
				gen = self.decode(q_gen)
				logger.add('images', 'gen', gen)
			except NotImplementedError:
				pass


	def forward(self, q):
		return self.decode(q)

	def regularize(self, q):
		return torch.tensor(0., device=q.device, dtype=q.dtype)

	def _step(self, batch):

		out = util.TensorDict()

		idx, (x, _) = batch

		q = self.retrieve(idx)
		pred = self.decode(q)

		loss = self.criterion(pred, x)
		total = loss
		if self.beta > 0:
			reg = self.regularize(q)
			total += self.beta * reg

			self.stats.update('reg', reg.detach())

			out['reg'] = reg.detach()

		self.optim_step(total)

		out.update({
			'loss': loss.detach(),
			'total': total.detach(),
			'original': x,
			'reconstruction': pred.detach(),
			'latent': q.detach(),
		})

		return out

	def retrieve(self, idx):
		return self.table(idx)

	def decode(self, q):
		return self.dec.decode(q)

	def sample_prior(self, N=1):
		raise NotImplementedError

	def generate(self, N=1):
		q = self.sample_prior(N)
		return self.decode(q)


class UniformDDecoder(DirectDecoder):

	def __init__(self, *args, cut_reset=False, repulsion_pow=2, **kwargs):
		super().__init__(*args, **kwargs)

		self.cut_reset = False
		self.repulsion_pow = repulsion_pow

	def sample_prior(self, N=1):
		q = torch.rand(N, self.latent_dim).to(self.table.weight.device).mul(2).sub(1)
		return q

	def reset(self):
		super().reset()

		space = self.table.weight.data

		if self.cut_reset:
			space[space > 1] = 1
			space[space < -1] = -1
		else:
			mx = space.max(-1)[0]
			sel = mx > 1
			space[sel, :] /= mx[sel].unsqueeze(-1)

			mn = space.min(-1)[0]
			sel = mn < -1
			space[sel, :] /= mn[sel].abs().unsqueeze(-1)

	def regularize(self, q):
		dists = F.pdist(q).pow(self.repulsion_pow)
		f = 1 / dists[dists > 0]
		if len(f) == 0:
			return torch.tensor(0., dtype=f.dtype, device=f.device)
		return f.mean()

class GaussianDDecoder(DirectDecoder):

	def sample_prior(self, N=1):
		q = torch.randn(N, self.latent_dim).to(self.table.weight.device)
		return q

	def regularize(self, q):
		# return q.norm(p=2)
		return q.pow(2).mean()

class SphericalDDecoder(DirectDecoder):

	def decode(self, q):
		q = self.normalize(q)
		return super().decode(q)

	def normalize(self, q):
		return F.normalize(q, p=2, dim=1)

	def sample_prior(self, N=1):
		q = torch.randn(N, self.latent_dim).to(self.table.weight.device)
		return q

	def regularize(self, q):
		q = self.normalize(q)
		dots = q @ q.T
		return dots.mean()


def get_model(args):
	if not hasattr(args, 'train_size'):
		args.train_size = 60000

	if args.decoder == 'conv':

		dec = models.Decoder(out_shape=args.out_shape, latent_dim=args.latent_dim, nonlin=args.nonlin, output_nonlin='sigmoid',
							  channels=args.channels, kernels=args.kernels, ups=args.factors,
							  upsampling=args.upsampling, norm_type=args.norm_type,
							  output_norm_type=None,
							  hidden_fc=args.fc,)

	elif args.decoder == 'fc':

		dec = MLP_Decoder(latent_dim=args.latent_dim, out_shape=args.out_shape, nonlin=args.nonlin,
						  hidden_dims=args.fc, output_nonlin='sigmoid')

	else:
		raise Exception('unknown {}'.format(args.decoder))

	print('Using a {} decoder'.format(args.decoder))


	ddec_args = {
		'decoder': dec,
		'latent_dim': args.latent_dim,
		'vocab_size': args.train_size,
		'zero_embedding': args.zero_embedding,
		'beta': args.beta,

	}

	if args.beta == 0:
		print('No regularization')

	if args.distr == 'gauss':
		model = GaussianDDecoder(**ddec_args)
	elif args.distr == 'uniform':
		model = UniformDDecoder(cut_reset=args.cut_reset, **ddec_args)
	elif args.distr == 'sphere':
		model = SphericalDDecoder(**ddec_args)
	elif args.distr == 'none':
		model = DirectDecoder(**ddec_args)
	else:
		raise Exception('unknown distr: {}'.format(args.distr))

	print('Using a distribution: {}'.format(args.distr))

	model.set_optim(optim_type=args.optim_type, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
	# optim = util.get_optimizer('sgd', model.parameters(), )

	model.to(args.device)

	if isinstance(model, fd.Schedulable) and args.decay_epochs > 0 and args.decay_factor > 0:
		model.scheduler = torch.optim.lr_scheduler.StepLR(model.optim,
													step_size=args.decay_epochs,
													gamma=args.decay_factor)

		print('Set scheduler: decay-epochs={}, decay-factor={}'.format(args.decay_epochs, args.decay_factor))

	return model



def main():
	raise Exception('deprecated')
	args = util.NS()

	args.device = 'cuda:0'
	args.seed = 0

	args.logdate = True
	args.tblog = True
	args.txtlog = False
	args.saveroot = '../trained_nets'

	args.dataset = 'mnist'
	args.indexed = True
	args.use_val = False

	args.num_workers = 0#4
	args.batch_size = 128

	args.start_epoch = 0
	args.epochs = 10

	args.name = 'direct_decoder'
	args.latent_dim = 3

	args.save_model = False

	now = time.strftime("%y-%m-%d-%H%M%S")
	if args.logdate:
		args.name = os.path.join(args.name, now)
	args.save_dir = os.path.join(args.saveroot, args.name)
	print('Save dir: {}'.format(args.save_dir))

	if args.tblog or args.txtlog:
		util.create_dir(args.save_dir)
		print('Logging in {}'.format(args.save_dir))
	logger = util.Logger(args.save_dir, tensorboard=args.tblog, txt=args.txtlog)

	# Set seed
	if not hasattr(args, 'seed') or args.seed is None:
		args.seed = util.get_random_seed()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	try:
		torch.cuda.manual_seed(args.seed)
	except:
		pass

	if not torch.cuda.is_available():
		args.device = 'cpu'
	print('Using device {} - random seed set to {}'.format(args.device, args.seed))

	datasets = train.load_data(args=args)

	loaders = train.get_loader(*datasets, batch_size=args.batch_size, num_workers=args.num_workers,
							   shuffle=True, drop_last=False, )

	trainloader, testloader = loaders[0], loaders[-1]
	valloader = None if len(loaders) == 2 else loaders[1]

	print('traindata len={}, trainloader len={}'.format(len(datasets[0]), len(trainloader)))
	if valloader is not None:
		print('valdata len={}, valloader len={}'.format(len(datasets[1]), len(valloader)))
	print('testdata len={}, testloader len={}'.format(len(datasets[-1]), len(testloader)))
	print('Batch size: {} samples'.format(args.batch_size))

	# Define Model
	args.total_samples = {'train': 0, 'test': 0}
	epoch = 0
	best_loss = None
	all_train_stats = []
	all_test_stats = []

	model = DirectDecoder(latent_dim=args.latent_dim, out_shape=(1, 28, 28), vocab_size=len(datasets[0]),

						  nonlin='prelu', output_nonlin='sigmoid',
						  channels=[32, 16, 8], kernels=[3, 3, 3], ups=[2, 2, 2], upsampling='bilinear',
						  output_norm_type=None,
						  hidden_fc=[128],
						  )

	model.to(args.device)
	print(model)
	print('Model has {} parameters'.format(util.count_parameters(model)))

	model.set_optim(optim_type='adam', lr=1e-3, weight_decay=1e-4, momentum=0.9)
	# optim = util.get_optimizer('sgd', model.parameters(), )
	scheduler = None  # torch.optim.lr_scheduler.StepLR(optim, step_size=6, gamma=0.2)

	lr = model.optim.param_groups[0]['lr']
	for _ in range(10):

		old_lr = lr
		if scheduler is not None:
			scheduler.step()
		lr = model.optim.param_groups[0]['lr']

		if lr != old_lr:
			print('--- lr update: {:.3E} -> {:.3E} ---'.format(old_lr, lr))

		train_stats = util.StatsMeter('lr', tau=0.1)
		train_stats.update('lr', lr)

		train_stats = train.run_epoch(model, trainloader, args, mode='train',
									  epoch=epoch, print_freq=10, logger=logger, silent=True,
									  viz_criterion_args=['reconstruction', 'original'],
									  stats=train_stats)

		all_train_stats.append(train_stats)

		print('[ {} ] Epoch {} Train={:.3f} ({:.3f})'.format(
			time.strftime("%H:%M:%S"), epoch + 1,
			train_stats['loss-viz'].avg.item(), train_stats['loss'].avg.item(),
		))

		if args.save_model:

			av_loss = train_stats['loss'].avg.item()
			is_best = best_loss is None or av_loss < best_loss
			if is_best:
				best_loss = av_loss

			path = train.save_checkpoint({
				'epoch': epoch,
				'args': args,
				'model_str': str(model),
				'model_state': model.state_dict(),
				'all_train_stats': all_train_stats,
				'all_test_stats': all_test_stats,

			}, args.save_dir, is_best=is_best, epoch=epoch + 1)
			print('--- checkpoint saved to {} ---'.format(path))

		epoch += 1

	print('Done')

	print('test complete.')

if __name__ == '__main__':

	main()
