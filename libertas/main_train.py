


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

from langimg import *



def main(argv=None):
	if argv is None:
		argv = sys.argv[1:]

	parser = train.setup_standard_options()

	###################
	# Additional options
	###################

	# Model
	parser.add_argument('--decoder', type=str, default='conv')

	parser.add_argument('--latent-dim', type=int, default=3)

	parser.add_argument('--normalize-latent', action='store_true')
	parser.add_argument('--zero-embedding', action='store_true')

	args = parser.parse_args(argv)

	if len(args.kernels) != len(args.channels):
		args.kernels = args.kernels*len(args.channels)

	if  len(args.factors) != len(args.channels):
		args.factors = args.factors * len(args.channels)

	###################
	# Logging
	###################

	now = time.strftime("%y-%m-%d-%H%M%S")
	if args.logdate:
		args.name += '_' + now
	args.save_dir = os.path.join(args.saveroot, args.name)
	# args.save_dir = os.path.join(args.save_root, args.name)
	print('Save dir: {}'.format(args.save_dir))
	if args.tblog or args.txtlog:
		util.create_dir(args.save_dir)
		print('Logging in {}'.format(args.save_dir))

	logger = util.Logger(args.save_dir, tensorboard=args.tblog, txt=args.txtlog)

	if args.no_cuda or not torch.cuda.is_available():
		args.device = 'cpu'
	print('Using {}'.format(args.device))

	###################
	# Seed
	###################

	# Set seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	try:
		torch.cuda.manual_seed(args.seed)
	except:
		pass

	###################
	# Data
	###################

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

	###################
	# Model
	###################

	args.total_samples = {'train': 0, 'test': 0}
	epoch = 0
	best_loss = None
	all_train_stats = []
	all_test_stats = []

	args.out_shape = (1, 28, 28)

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


	model = DirectDecoder(decoder=dec, latent_dim=args.latent_dim, vocab_size=len(datasets[0]),

	                      normalize_latent=args.normalize_latent, zero_embedding=args.zero_embedding,


	                      )

	model.set_optim(optim_type=args.optim_type, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
	# optim = util.get_optimizer('sgd', model.parameters(), )
	scheduler = None  # torch.optim.lr_scheduler.StepLR(optim, step_size=6, gamma=0.2)

	model.to(args.device)
	print(model)
	print('Model has {} parameters'.format(util.count_parameters(model)))

	scheduler = None
	if args.decay_epochs > 0 and args.decay_factor > 0:
		scheduler = torch.optim.lr_scheduler.StepLR(model.optim,
		                                            step_size=args.decay_epochs,
		                                            gamma=args.decay_factor)

	lr = model.optim.param_groups[0]['lr']

	###################
	# Reseed after model init
	###################

	# Set seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	try:
		torch.cuda.manual_seed(args.seed)
	except:
		pass

	###################
	# Run Train/Val Epochs
	###################

	if args.no_test:
		print('Will not run test data after training')

	epoch = args.start_epoch
	for ep in range(args.epochs):

		old_lr = lr
		if scheduler is not None:
			scheduler.step()
		lr = model.optim.param_groups[0]['lr']

		if lr != old_lr:
			print('--- lr update: {:.3E} -> {:.3E} ---'.format(old_lr, lr))

		train_stats = util.StatsMeter('lr', tau=0.1)
		train_stats.update('lr', lr)

		train_stats = train.run_epoch(model, trainloader, args, mode='train',
		                              epoch=epoch, print_freq=args.print_freq, logger=logger, silent=False,
		                              viz_criterion_args=['reconstruction', 'original'],
		                              stats=train_stats, )

		all_train_stats.append(train_stats)

		# print('[ {} ] Epoch {} Train={:.3f} ({:.3f})'.format(
		# 	time.strftime("%H:%M:%S"), epoch + 1,
		# 	train_stats['loss-viz'].avg.item(), train_stats['loss'].avg.item(),
		# ))

		if epoch % args.save_freq == 0:
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
		print()


	###################
	# Run Test Epochs
	###################













if __name__ == '__main__':
	argv = ['--config', 'config/simple.yaml', '--name', 'test-ddecoder']

	argv = None

	main(argv)


