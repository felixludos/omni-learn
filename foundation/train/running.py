
import sys, os, time
import traceback, ipdb
from tqdm import tqdm
import yaml
import torch
from .. import util
from ..framework import Visualizable, Recordable
# from .load_data import old_get_loaders as get_loaders # TODO: update

from .setup import setup_records, setup_logging
from .loading import load, save_checkpoint
from .data import get_loaders



def run_full(A, get_data, get_model, get_name=None):

	if 'device' not in A or not torch.cuda.is_available():
		A.device = 'cpu'
	print('Using device: {}'.format(A.device))

	# Set seed
	if 'seed' not in A:
		A.seed = util.gen_random_seed()
	print('Using pegasus seed: {}'.format(A.seed))
	
	###################
	# Data/Model
	###################

	path, extend = None, None
	if 'resume' in A:
		path = A.resume
		assert not os.path.isfile(path), 'When resuming you should only specify the directory of the run, not the specific checkpoint'
		if 'extend' in A:
			extend = A.extend
		A.clear()
	if 'load' in A:
		path = A.load
		A.loaded = A.load
	
	A, (*datasets, testset), model, ckpt = load(path=path, A=A, mode='train',
	                                            load_last=A is None, # load will load the best, rather than last
	                                              get_model=get_model, get_data=get_data,
	                                              return_args=True, return_ckpt=True)

	###################
	# Logging
	###################

	if 'name' not in A:
		A.name = get_name(A) if get_name is not None else None

	logger = setup_logging(A.output)

	if ckpt is None: # novel
		records = setup_records(A.training)
		epoch_seed = util.gen_deterministic_seed(A.seed)
	else: # resume, load, complete
		records = ckpt['records']
		epoch_seed = ckpt['epoch_seed']
		if 'load' in A: # load
			del A.load
			print('WARNING: you are loading a previous model!')
		else: # resume, complete
			A.training.start = records['epoch']
			if extend is not None: # resume
				A.training.epochs += extend
				print('Extending training by {} epochs'.format(extend))
			print('Running {} more epochs'.format(A.training.epochs))

	if 'save_freq' not in A.output or 'save_dir' not in A.output:
		A.output.save_freq = -1
	if 'track_best' not in A.training:
		A.training.track_best = False

	if 'save_dir' in A.output and ('config.tml' not in os.listdir(A.output.save_dir) or extend is not None): # new save_dir - novel, load
		config_path = A.export(os.path.join(A.output.save_dir, 'config.tml'))
		print('Config saved to {}'.format(config_path))

	if 'save_dir' in A.output and os.environ['FOUNDATION_RUN_MODE'] == 'cluster' and 'JOBDIR' in os.environ: # cluster checkpointing for restarts

		jobdir = os.environ['JOBDIR']

		if 'checkpoints.txt' not in os.listdir(jobdir):

			# register job
			if 'JOB_ID' in os.environ:
				with open(os.environ['JOB_REGISTRY_PATH'], 'a+') as f:
					f.write('{:>10} - {}\n'.format(os.environ['JOB_ID'].split('#')[-1], A.output.save_dir))

			with open(os.path.join(jobdir, 'checkpoints.txt'), 'w') as f:
				f.write(os.path.basename(A.output.save_dir))
			print('[Saved checkpoint dir for restarts]')

	###################
	# DataLoader
	###################

	assert 'batch_size' in A.dataset, 'No batch_size found'

	if 'shuffle' not in A.dataset:
		A.dataset.shuffle = True
	if 'drop_last' not in A.dataset:
		A.dataset.drop_last = True

	loaders = get_loaders(*datasets, batch_size=A.dataset.batch_size, num_workers=A.num_workers,
	                                            shuffle=A.dataset.shuffle, drop_last=A.dataset.drop_last, silent=True)

	# TODO: generalize for other validation methods

	if len(datasets) > 1:
		trainloader, *valloaders = loaders
	else:
		trainloader = loaders
		valloaders = []
	assert len(valloaders) < 2, 'Multiple validation sets are not supported (yet)'
	valloader = valloaders[0] if len(valloaders) else None

	print('traindata len={}, trainloader len={}'.format(len(datasets[0]), len(trainloader)))
	if valloader is not None:
		print('valdata len={}, valloader len={}'.format(len(datasets[1]), len(valloader)))
	if testset is not None:
		print('testdata len={}'.format(len(testset[-1])))
	else:
		print('testdata not loaded yet')
	print('Batch size: {} samples'.format(A.dataset.batch_size))

	if 'stats_decay' not in A.output:
		A.output.stats_decay = max(0.01, min(100/len(trainloader), 0.1))
	util.set_default_tau(A.output.stats_decay)
	if isinstance(model, Recordable):
		model.stats.set_tau(A.output.stats_decay)
	if 'print_freq' not in A.output:
		A.output.print_freq = min(max(20, len(trainloader) // 40), 200)


	###################
	# Model
	###################

	print(model)
	print(model.optim)
	print('Model has {} parameters'.format(util.count_parameters(model)))

	###################
	# Run Train/Val Epochs
	###################

	for i in range(A.training.epochs - A.training.start):
		is_best = False

		records['epoch'] += 1

		util.set_seed(epoch_seed)

		train_stats = run_epoch(model, trainloader, A, mode='train', records=records,
		                              logger=logger, silent=False, inline='inline' in A and A.inline)

		records['stats']['train'].append(train_stats.export())

		if valloader is not None:
			val_stats = run_epoch(model, valloader, A, mode='val', records=records,
		                              logger=logger, silent=False, inline='inline' in A and A.inline)

			records['stats']['val'].append(val_stats.export())

		epoch_seed = util.gen_deterministic_seed(A.seed)

		if A.output.save_freq > 0 and (records['epoch']-1) % A.output.save_freq == 0:
			ckpt = {
				'model_str': str(model),
				'model_state': model.state_dict(),

				'records': records,
				'epoch_seed': epoch_seed,
			}

			loss = train_stats['loss'] if valloader is None else val_stats['loss']

			if A.training.track_best:
				if loss.count == 0:
					print('WARNING: loss has not been logged, so the best model cant be tracked')
				else:
					av_loss = loss.avg.item()
					is_best = records['best']['loss'] is None or av_loss < records['best']['loss']
					if is_best:
						if records['best']['loss'] is not None:
							print('Epoch {} improved from {:.4f} (epoch={}) to: {:.4f}'.format(
								records['epoch'], records['best']['loss'], records['best']['epoch'], av_loss))
						records['best']['loss'] = av_loss
						records['best']['epoch'] = records['epoch']


			path = save_checkpoint(ckpt, A.output.save_dir, is_best=is_best, epoch=records['epoch'])
			print('--- checkpoint saved to {} ---'.format(path))

		print()

		# print('*** Exiting with status 3')
		# sys.exit(3)

	###################
	# Run Test Epochs
	###################

	if 'no_test' not in A.training or not A.training.no_test:

		records['test_epoch'] = records['epoch']

		if A.training.track_best and 'save_dir' in A.output:
			model, ckpt = load(path=A.output.save_dir, mode='test', get_model=get_model,
			                return_args=False, return_ckpt=True)
			print('Loaded best model, trained for {} epochs'.format(ckpt['records']['epoch']))

			records['test_epoch'] = ckpt['records']['epoch']


		if testset is None:
			testset = get_data(A, mode='test')

		testloader = get_loaders(testset, batch_size=A.dataset.batch_size, num_workers=A.num_workers,
                                            shuffle=A.dataset.shuffle, drop_last=A.dataset.drop_last, silent=True)

		print('testdata len={}, testloader len={}'.format(len(testset), len(testloader)))

		test_stats = run_epoch(model, testloader, A, records=records, mode='test',
		                      logger=logger, silent=False, inline='inline' in A and A.inline)

		records['stats']['test'] = test_stats.export()

		results_path = os.path.join(A.output.save_dir, 'results.yaml')
		with open(results_path, 'w') as f:
			yaml.dump(records, f)
		print('Final results saved to {}'.format(results_path))


	return model, datasets, loaders


def run_epoch(model, loader, A, records, mode='test',
                logger=None, unique_tests=False, silent=False,
                stats=None, inline=False):
	train = mode == 'train'
	if train:
		model.train()
	else:
		model.eval()
		
	total_epochs = A.training.epochs #+ A.training.start
	print_freq = A.output.print_freq
	
	viz_criterion = None
	if 'viz_criterion' in A.training and 'viz_criterion_args' in A.training:
		viz_criterion = util.get_loss_type(A.training.viz_criterion)
		viz_criterion_args = A.training.viz_criterion_args
	
	model.pre_epoch()

	if stats is None:
		stats = util.StatsMeter('loss')
		if isinstance(model, Recordable):
			stats.shallow_join(model.stats)
	elif 'loss' not in stats:
		stats.new('loss')
	if viz_criterion is not None and 'loss-viz' not in stats:
		stats.new('loss-viz')

	time_stats = util.StatsMeter('data', 'model', 'viz')
	stats.shallow_join(time_stats, fmt='time-{}')

	logger_prefix = '{}/{}'.format('{}',mode) if not unique_tests or train else '{}/{}{}'.format('{}',
		mode, records['epoch'])


	itr = enumerate(iter(loader))
	if inline:
		itr = tqdm(itr, total=len(loader), leave=True)

	start = time.time()
	for i, batch in itr:
		batch = util.to(batch, A.device)

		B = batch.size(0)

		records['total_samples'][mode] += B

		time_stats.update('data', time.time() - start)
		start = time.time()

		if train:
			out = model.step(batch)
		else:
			out = model.test(batch)
		if 'loss' in out:
			stats.update('loss', out.loss.detach())

		time_stats.update('model', time.time() - start)
		start = time.time()

		if logger is not None and print_freq is not None and i % print_freq == 0:

			logger.set_step(records['total_samples'][mode])
			logger.set_tag_format(logger_prefix)

			with torch.no_grad():
				if viz_criterion is not None:
					stats.update('loss-viz', viz_criterion(*[out[n] for n in viz_criterion_args]).detach())

				if logger is not None:
					for k,v in stats.smooths().items():
						logger.add('scalar', k, v)

				if isinstance(model, Visualizable):
					model.visualize(out, logger)

		if not silent:
			loss_info = ' Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
				if stats['loss'].count > 0 else ''
			if inline:
				itr.set_description('{} {}/{}{}'.format(mode, records['epoch'], total_epochs, loss_info))
			elif print_freq is None or i % print_freq == 0:
				print('[ {} ] {} Ep={}/{} Itr={}/{}{}'.format(
					time.strftime("%H:%M:%S"), mode,
					records['epoch'], total_epochs, i + 1, len(loader), loss_info))

				sys.stdout.flush()

		time_stats.update('viz', time.time() - start)
		start = time.time()

	if not silent:
		loss_info = ' Loss: {:.4f} ({:.4f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
			if stats['loss'].count > 0 else ''
		msg = '[ {} ] {} Ep={}/{} complete{}'.format(
			time.strftime("%H:%M:%S"), mode,
			records['epoch'], total_epochs, loss_info)
		border = '-' * 50
		print(border)
		print(msg)
		print(border)

	if logger is not None:
		logger.flush()
	sys.stdout.flush()

	return stats










