
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torchvision
from ..framework import Generative, Recordable
from .. import util
from ..framework import Visualizable
from .load_data import get_loaders
from .options import setup_standard_options
from .. import models
from .load_model import save_checkpoint

def run_full(get_options, get_data, get_model, argv=None):
	if argv is None:
		argv = sys.argv[1:]

	parser = get_options() # shuffle, drop_last, viz_criterion_args, track_best
	parser = setup_standard_options(parser)

	# Model

	args = parser.parse_args(argv)

	if hasattr(args, 'kernels') and args.kernels is not None:
		if len(args.kernels) != len(args.channels):
			args.kernels = args.kernels * len(args.channels)

	if hasattr(args, 'factors') and args.kernels is not None:
		if len(args.factors) != len(args.channels):
			args.factors = args.factors * len(args.channels)

	###################
	# Logging
	###################

	now = time.strftime("%y%m%d-%H%M%S")
	if args.logdate:
		args.name = args.name + '_' + now
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

	assert not args.use_val, 'cant use validation'

	datasets = get_data(args)

	loaders = get_loaders(*datasets, batch_size=args.batch_size, num_workers=args.num_workers,
	                                            shuffle=args.shuffle, drop_last=args.drop_last, silent=True)

	trainloader = loaders[0]
	testloader = None if len(loaders) < 2 else loaders[-1]
	valloader = None if len(loaders) < 3 else loaders[1]

	print('traindata len={}, trainloader len={}'.format(len(datasets[0]), len(trainloader)))
	if valloader is not None:
		print('valdata len={}, valloader len={}'.format(len(datasets[1]), len(valloader)))
	if testloader is not None:
		print('testdata len={}, testloader len={}'.format(len(datasets[-1]), len(testloader)))
	print('Batch size: {} samples'.format(args.batch_size))

	# Reseed after loading datasets
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	try:
		torch.cuda.manual_seed(args.seed)
	except:
		pass

	###################
	# Model
	###################

	args.total_samples = {'train': 0, 'test': 0}
	if valloader is not None:
		args.total_samples['val'] = 0
	epoch = args.start_epoch
	best_loss = None
	best_epoch = None
	is_best = False
	all_train_stats = []
	all_val_stats = []

	if args.stats_decay is None:
		args.stats_decay = max(min(1 / len(trainloader), 0.1), 0.01)
	util.set_default_tau(args.stats_decay)

	model = get_model(args)

	model.to(args.device)

	print(model)
	print(model.optim)
	print('Model has {} parameters'.format(util.count_parameters(model)))

	# Reseed after model init
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
	else:
		raise NotImplementedError


	for _ in range(args.epochs):

		model.reset()

		train_stats = util.StatsMeter()
		train_stats.shallow_join(model.stats)

		train_stats = run_epoch(model, trainloader, args, mode='train',
		                              epoch=epoch, print_freq=args.print_freq, logger=logger, silent=False,
		                              viz_criterion_args=args.viz_criterion_args,
		                              stats=train_stats, )

		all_train_stats.append(train_stats.copy())

		if valloader is not None:
			model.reset()

			val_stats = util.StatsMeter()
			val_stats.shallow_join(model.stats)

			val_stats = run_epoch(model, valloader, args, mode='val',
		                              epoch=epoch, print_freq=args.print_freq, logger=logger, silent=False,
		                              viz_criterion_args=args.viz_criterion_args,
		                              stats=val_stats, )

			all_val_stats.append(val_stats.copy())



		if args.save_freq > 0 and epoch % args.save_freq == 0:


			ckpt = {
				'epoch': epoch+1,

				'args': args,

				'model_str': str(model),
				'model_state': model.state_dict(),
				'all_train_stats': all_train_stats,
			}
			if args.track_best:
				av_loss = train_stats['loss'].avg.item() if valloader is None else val_stats['loss'].avg.item()
				is_best = best_loss is None or av_loss < best_loss
				if is_best:
					best_loss = av_loss
					best_epoch = epoch

				ckpt['loss'] = av_loss
				ckpt['best_loss'] = best_loss
				ckpt['best_epoch'] = best_epoch
			if len(all_val_stats):
				ckpt['all_val_stats'] = all_val_stats
			path = save_checkpoint(ckpt, args.save_dir, is_best=is_best, epoch=epoch+1)
			print('--- checkpoint saved to {} ---'.format(path))

		epoch += 1
		print()

	###################
	# Run Test Epochs
	###################

	print('No test epoch implemented')

	return model, datasets, loaders


def run_epoch(model, loader, args, mode='test',
                epoch=None, print_freq=-1, img_freq=-1, logger=None, unique_tests=False, silent=False,
                stats=None, viz_criterion='mse', viz_criterion_args=None):
	train = mode == 'train'
	if train:
		model.train()
	else:
		model.eval()
	if not hasattr(args, 'total_samples'):
		args.total_samples = {'train': 0, 'val':0, 'test': 0}

	viz_criterion = util.get_loss_type(viz_criterion) if viz_criterion_args is not None else None

	print_freq = max(1, len(loader) // 100) if print_freq < 0 else print_freq
	img_freq = print_freq*5 if img_freq < 0 and print_freq is not None else img_freq
	if print_freq == 0:
		print_freq = None
	if img_freq == 0:
		img_freq = None

	if stats is None:
		stats = util.StatsMeter('loss', tau=max(min(1 / len(loader), 0.1), 0.01))
	elif 'loss' not in stats:
		stats.new('loss')

	if viz_criterion is not None and 'loss-viz' not in stats:
		stats.new('loss-viz')

	time_stats = util.StatsMeter('data', 'model', 'viz', tau=args.stats_decay)
	stats.shallow_join(time_stats, fmt='time-{}')

	logger_prefix = '{}/{}'.format('{}',mode) if not unique_tests or train else '{}/{}{}'.format('{}',
		mode, epoch + 1)

	itr = iter(loader)
	start = time.time()
	for i, batch in enumerate(itr):
		batch = util.to(batch, args.device)

		B = batch.size(0)

		args.total_samples[mode] += B

		time_stats.update('data', time.time() - start)
		start = time.time()

		if train:
			out = model.step(batch)
		else:
			out = model.test(batch)
		stats.update('loss', out.loss.detach())

		time_stats.update('model', time.time() - start)
		start = time.time()

		if logger is not None and print_freq is not None and i % print_freq == 0:

			logger.set_step(args.total_samples[mode])
			logger.set_tag_format(logger_prefix)

			with torch.no_grad():
				if viz_criterion is not None:
					stats.update('loss-viz', viz_criterion(*[out[n] for n in viz_criterion_args]).detach())

				for k,v in stats.smooths().items():
					logger.add('scalar', k, v)

				if isinstance(model, Visualizable):
					model.visualize(out, logger)

			if not silent:
				print('[ {} ] {} Ep={}/{} Itr={}/{} Loss: {:.3f} ({:.3f})'.format(
					time.strftime("%H:%M:%S"), mode,
					epoch + 1, args.epochs, i + 1, len(loader),
					stats['loss'].val.item(), stats['loss'].smooth.item()))

		time_stats.update('viz', time.time() - start)
		start = time.time()

	if not silent:
		msg = '[ {} ] {} Ep={}/{} complete Loss: {:.4f} ({:.4f})'.format(
			time.strftime("%H:%M:%S"), mode,
			epoch + 1, args.epochs,
			stats['loss'].val.item(), stats['loss'].smooth.item())
		border = '-' * 50
		print(border)
		print(msg)
		print(border)

	if logger is not None:
		logger.flush()

	return stats










