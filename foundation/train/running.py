
import sys, os, time
# import traceback, ipdb
from tqdm import tqdm
import yaml
import torch
from .. import util
from ..framework import Visualizable, Recordable, Schedulable
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

	sys.stdout.flush()

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
		A.loaded = path

	A, (*datasets, testset), model, ckpt = load(path=path, A=A, mode='train',
	                                            load_last=True, # load will load the best, rather than last
	                                            load_optim='load' not in A, load_scheduler='load' not in A, # only load network parameters when "loading" pretrained model
	                                              get_model=get_model, get_data=get_data,
	                                              return_args=True, return_ckpt=True, )#strict='load' not in A)

	###################
	# Logging
	###################

	if 'name' not in A:
		A.name = get_name(A) if get_name is not None else None

	if 'save_dir' in A.output: # TODO: is this necessary?
		del A.output.save_dir

	logger = setup_logging(A.output)

	if 'date' not in A.info and '_logged_date' in A.output:
		A.info.date = A.output._logged_date

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
			if extend is not None: # resume
				A.training.epochs = extend
				print('Extending training to {} epochs'.format(extend))
			print('Running {} more epochs'.format(A.training.epochs - records['epoch']))

	if 'save_freq' not in A.output or 'save_dir' not in A.output:
		A.output.save_freq = -1
	if 'track_best' not in A.training:
		A.training.track_best = False

	if 'save_dir' in A.output:
		if ('config.yml' not in os.listdir(A.output.save_dir) or extend is not None): # new save_dir - novel, load
			config_path = A.export(os.path.join(A.output.save_dir, 'config.yml'))
			print('Config saved to {}'.format(config_path))

		if os.environ['FOUNDATION_RUN_MODE'] == 'cluster' and 'JOBDIR' in os.environ: # cluster checkpointing for restarts

			jobdir = os.environ['JOBDIR']

			cname = 'checkpoints{}.txt'.format(os.environ['PROCESS_ID'])

			if cname not in os.listdir(jobdir):

				# register job
				if 'JOB_ID' in os.environ:
					with open(os.environ['JOB_REGISTRY_PATH'], 'a+') as f:
						f.write('{:<12} - {} - {}\n'.format(os.environ['JOB_ID'].split('#')[-1],
						                                    os.path.basename(A.output.save_dir),
						                                    os.path.basename(jobdir)))

				with open(os.path.join(jobdir, cname), 'w') as f:
					f.write(os.path.basename(A.output.save_dir))
				print('[Saved checkpoint dir for restarts]')

	if 'RESTART_AFTER' in os.environ:
		print('Will restart after {} epochs.'.format(os.environ['RESTART_AFTER']))

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
	elif isinstance(model, Schedulable):
		assert model.scheduler is None or not model.scheduler.req_loss, \
			'no validation set, but lr scheduler requires loss'
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
		print('Print freq set to: {}'.format(A.output.print_freq))
	if 'unique_tests' not in A.output:
		A.output.unique_tests = False
		print('Validation of each epoch is not logged separately')


	###################
	# Model
	###################

	print(model)
	print(model.optim)
	if hasattr(model, 'scheduler'):
		print(model.scheduler)
	print('Model has {} parameters'.format(util.count_parameters(model)))

	sys.stdout.flush()

	###################
	# Run Train/Val Epochs
	###################

	_restart_counter = 0
	N = A.training.epochs - records['epoch']

	for i in range(N):
		is_best = False

		records['epoch'] += 1

		util.set_seed(epoch_seed)

		model.pre_epoch(mode='train', epoch=records['epoch'])

		train_stats = run_epoch(model, trainloader, A, mode='train', records=records,
		                              logger=logger, silent=False, inline='inline' in A and A.inline)

		model.post_epoch(mode='train', epoch=records['epoch'], stats=train_stats)

		records['stats']['train'].append(train_stats.export())

		if valloader is not None:
			model.pre_epoch(mode='val', epoch=records['epoch'])

			val_stats = run_epoch(model, valloader, A, mode='val', records=records, unique_tests=A.output.unique_tests,
		                              logger=logger, silent=False, inline='inline' in A and A.inline)

			records['stats']['val'].append(val_stats.export())

			model.post_epoch(mode='val', epoch=records['epoch'], stats=val_stats)

		epoch_seed = util.gen_deterministic_seed(A.seed)

		if A.output.save_freq > 0 and ((records['epoch']-1) % A.output.save_freq == 0
		                               or records['epoch'] == A.training.epochs):
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

			_restart_counter += 1

		print()

		if 'RESTART_AFTER' in os.environ and ((_restart_counter+1) % int(os.environ['RESTART_AFTER']) == 0) and (N > i+1):

			print('*** Exiting for restart after {} checkpoints'.format(os.environ['RESTART_AFTER']))
			sys.exit(3)

	print('Training complete.')

	return model, datasets, loaders, records

	###################
	# Run Test Epochs
	###################

	if 'no_test' not in A.training or not A.training.no_test:

		records['test_epoch'] = records['epoch']

		if A.training.track_best and 'save_dir' in A.output:
			try:
				model, ckpt = load(path=A.output.save_dir, mode='test', get_model=get_model, get_data=None,
				                return_args=False, return_ckpt=True, force_load_model=True)
				print('Loaded best model, trained for {} epochs'.format(ckpt['records']['epoch']))
				records['test_epoch'] = ckpt['records']['epoch']
			except FileNotFoundError:
				print('Using current model for testing')

		if testset is None:
			testset = get_data(A, mode='test')

		testloader = get_loaders(testset, batch_size=A.dataset.batch_size, num_workers=A.num_workers,
                                            shuffle=A.dataset.shuffle, drop_last=A.dataset.drop_last, silent=True)

		print('testdata len={}, testloader len={}'.format(len(testset), len(testloader)))

		test_stats = run_epoch(model, testloader, A, records=records, mode='test',
		                      logger=logger, silent=False, inline='inline' in A and A.inline)

		records['stats']['test'] = test_stats.export()

		if 'save_dir' in A.output:
			results_path = os.path.join(A.output.save_dir, 'results.yaml')
			with open(results_path, 'w') as f:
				yaml.dump(records, f)
			print('Final results saved to {}'.format(results_path))


	return model, datasets, loaders, records


def run_epoch(model, loader, A, records, mode='test',
              logger=None, unique_tests=False, silent=False,
              stats=None, inline=False):
	train = mode == 'train'
	if train:
		model.train()
	else:
		model.eval()

	epoch = records['epoch']
	total_epochs = A.training.epochs  # + A.training.start
	print_freq = A.output.print_freq

	viz_criterion = None
	if 'viz_criterion' in A.training and 'viz_criterion_args' in A.training:
		viz_criterion = util.get_loss_type(A.training.viz_criterion)
		viz_criterion_args = A.training.viz_criterion_args

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

	logger_prefix = '{}/{}'.format('{}', mode) if not unique_tests or train else 'zz{}-{}/{}'.format(epoch, '{}',
	                                                                                                 mode)

	total = len(loader)

	max_iter = A.training.max_iter if 'max_iter' in A.training else None
	if max_iter is not None:
		print('\n\nWARNING: not running full epochs\n\n')
		total = max_iter

	itr = enumerate(iter(loader))
	if inline:
		itr = tqdm(itr, total=total, leave=True)

	start = time.time()
	for i, batch in itr:

		if max_iter is not None and i == max_iter:
			print('WARNING: ending epoch prematurely! - hit max_iter')
			del itr
			break

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
					for k, v in stats.smooths().items():
						logger.add('scalar', k, v)

				if isinstance(model, Visualizable):
					model.visualize(out, logger)

		if not silent:
			loss_info = ' Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
				if stats['loss'].count > 0 else ''
			if inline:
				itr.set_description('{} {}/{}{}'.format(mode, epoch, total_epochs, loss_info))
			elif print_freq is None or i % print_freq == 0:
				print('[ {} ] {} Ep={}/{} Itr={}/{}{}'.format(
					time.strftime("%H:%M:%S"), mode,
					epoch, total_epochs, i + 1, len(loader), loss_info))

				sys.stdout.flush()

		time_stats.update('viz', time.time() - start)
		start = time.time()

		del out
		torch.cuda.empty_cache()

	if not silent:
		loss_info = ' Loss: {:.4f} ({:.4f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
			if stats['loss'].count > 0 else ''
		msg = '[ {} ] {} Ep={}/{} complete{}'.format(
			time.strftime("%H:%M:%S"), mode,
			epoch, total_epochs, loss_info)
		border = '-' * len(msg)

		if inline:
			print()
		print(border)
		print(msg)
		print(border)

	if logger is not None:
		logger.flush()
	sys.stdout.flush()

	return stats



def new_run_full(A, get_data, get_model, get_name=None):
	## %%%%%%%%%%%%%%%%%%

	if 'device' not in A or not torch.cuda.is_available():
		A.device = 'cpu'
	print('Using device: {}'.format(A.device))

	# Set seed
	if 'seed' not in A:
		A.seed = util.gen_random_seed()
	print('Using pegasus seed: {}'.format(A.seed))

	sys.stdout.flush()


	#####################
	# region Create Data/Model
	#####################

	path, extend = None, None
	if 'resume' in A:
		path = A.resume
		assert not os.path.isfile(path), 'When resuming you should only specify the directory of the run, not the specific checkpoint'
		if 'extend' in A:
			extend = A.extend
		A.clear()
	if 'load' in A:
		path = A.load
		A.loaded = path

	A, (*datasets, testset), model, ckpt = load(path=path, A=A, mode='train',
	                                            load_last=True, # load will load the best, rather than last
	                                            load_optim='load' not in A, load_scheduler='load' not in A, # only load network parameters when "loading" pretrained model
	                                              get_model=get_model, get_data=get_data,
	                                              return_args=True, return_ckpt=True, )#strict='load' not in A)

	# endregion

	#####################
	# region Logging
	#####################

	if 'name' not in A:
		A.name = get_name(A) if get_name is not None else None

	if 'save_dir' in A.output: # TODO: is this necessary?
		del A.output.save_dir

	logger = setup_logging(A.output)

	if 'date' not in A.info and '_logged_date' in A.output:
		A.info.date = A.output._logged_date

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
			if extend is not None: # resume
				A.training.epochs = extend
				print('Extending training to {} epochs'.format(extend))
			print('Running {} more epochs'.format(A.training.epochs - records['epoch']))

	if 'save_freq' not in A.output or 'save_dir' not in A.output:
		A.output.save_freq = -1
	if 'track_best' not in A.training:
		A.training.track_best = False

	if 'save_dir' in A.output:
		if ('config.yml' not in os.listdir(A.output.save_dir) or extend is not None): # new save_dir - novel, load
			config_path = A.export(os.path.join(A.output.save_dir, 'config.yml'))
			print('Config saved to {}'.format(config_path))

		if os.environ['FOUNDATION_RUN_MODE'] == 'cluster' and 'JOBDIR' in os.environ: # cluster checkpointing for restarts

			jobdir = os.environ['JOBDIR']

			cname = 'checkpoints{}.txt'.format(os.environ['PROCESS_ID'])

			if cname not in os.listdir(jobdir):

				# register job
				if 'JOB_ID' in os.environ:
					with open(os.environ['JOB_REGISTRY_PATH'], 'a+') as f:
						f.write('{:<12} - {} - {}\n'.format(os.environ['JOB_ID'].split('#')[-1],
						                                    os.path.basename(A.output.save_dir),
						                                    os.path.basename(jobdir)))

				with open(os.path.join(jobdir, cname), 'w') as f:
					f.write(os.path.basename(A.output.save_dir))
				print('[Saved checkpoint dir for restarts]')

	# if 'RESTART_AFTER' in os.environ:
	# 	print('Will restart after {} epochs.'.format(os.environ['RESTART_AFTER']))




	# endregion

	#####################
	# region DataLoader
	#####################

	assert 'batch_size' in A.dataset, 'No batch_size found'

	if 'shuffle' not in A.dataset:
		A.dataset.shuffle = True
	if 'drop_last' not in A.dataset:
		A.dataset.drop_last = True

	loaders = get_loaders(*datasets, batch_size=A.dataset.batch_size, num_workers=A.num_workers,
	                                            shuffle=A.dataset.shuffle, drop_last=A.dataset.drop_last, silent=True)

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
	elif isinstance(model, Schedulable):
		assert model.scheduler is None or not model.scheduler.req_loss, \
			'no validation set, but lr scheduler requires loss'
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
		print('Print freq set to: {}'.format(A.output.print_freq))
	if 'unique_tests' not in A.output:
		A.output.unique_tests = False
		print('Validation of each epoch is not logged separately')


	# endregion

	#####################
	# region Model
	#####################


	print(model)
	print(model.optim)
	if hasattr(model, 'scheduler'):
		print(model.scheduler)
	print('Model has {} parameters'.format(util.count_parameters(model)))

	sys.stdout.flush()

	# endregion

	#####################
	# region Run Train/Val
	#####################

	edata = util.tdict()
	edata.epoch_seed = epoch_seed

	run_continuous(A, records, model, trainloader, valloader,
	               logger=None, unique_tests=False, silent=False,
                   stats=None, inline=False, epoch_seed=epoch_seed)


	print('Training complete.')

	return model, datasets, loaders, records

	# endregion

	#####################
	# region Run Test
	#####################

	if 'no_test' not in A.training or not A.training.no_test:

		records['test_epoch'] = records['epoch']

		if A.training.track_best and 'save_dir' in A.output:
			try:
				model, ckpt = load(path=A.output.save_dir, mode='test', get_model=get_model, get_data=None,
				                   return_args=False, return_ckpt=True, force_load_model=True)
				print('Loaded best model, trained for {} epochs'.format(ckpt['records']['epoch']))
				records['test_epoch'] = ckpt['records']['epoch']
			except FileNotFoundError:
				print('Using current model for testing')

		if testset is None:
			testset = get_data(A, mode='test')

		testloader = get_loaders(testset, batch_size=A.dataset.batch_size, num_workers=A.num_workers,
		                         shuffle=A.dataset.shuffle, drop_last=A.dataset.drop_last, silent=True)

		print('testdata len={}, testloader len={}'.format(len(testset), len(testloader)))

		test_stats = run_epoch(model, testloader, A, records=records, mode='test',
		                       logger=logger, silent=False, inline='inline' in A and A.inline)

		records['stats']['test'] = test_stats.export()

		if 'save_dir' in A.output:
			results_path = os.path.join(A.output.save_dir, 'results.yaml')
			with open(results_path, 'w') as f:
				yaml.dump(records, f)
			print('Final results saved to {}'.format(results_path))

	# endregion

	return model, datasets, loaders, records


def save_checkpoint(A, model, records, loss, N):
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


def restart_loader(loader, epoch_seed):

	util.set_seed(epoch_seed)
	loader = iter(loader)

	epoch_seed = util.gen_deterministic_seed(epoch_seed)

	return loader, epoch_seed

def gen_logger_prefix(mode, unique_test=False, epoch=None):

	# if unique_test and mode != 'train':
	# 	return 'zz{}-{}/{}'.format(epoch, '{}',mode)

	return '{}/{}'.format('{}', mode)


def logger_step(logger, stats, smooths=True):
	logger.set_step(records['total_samples']['train'] if display_samples else records['total_steps'])
	logger.set_tag_format('{}/train')

	with torch.no_grad():
		display = stats.smooths() if smooths else stats.avgs()
		for k, v in display.items():
			logger.add('scalar', k, v)




def run_continuous(A, records, model, trainloader, valloader=None,
                   logger=None, unique_tests=False, silent=False,
                   display_samples=False,
                   stats=None, inline=False, epoch_seed=None, pbar=None):

	# region Limits/Freqs

	if 'step_limit' not in A.training: # step limits are soft, sample limits are hard
		A.training.step_limit = A.training.epochs*len(trainloader)
		print('Step limit set to {} steps'.format(A.training.step_limit))

	if 'total_steps' not in records:
		records['total_steps'] = records['epoch']*len(trainloader)

	step_limit = A.training.step_limit
	sample_limit = A.training.sample_limit if 'sample_limit' in A.training else None

	save_freq = A.output.save_freq
	if save_freq < len(trainloader):
		print('WARNING: saving more than once per epoch: checkpoint every {} iterations'.format(save_freq))

		if 'quick_save' not in A.output or not A.output.quick_save:
			A.output.save_freq = len(trainloader)

	print_freq = A.output.print_freq
	if pbar is not None:
		print_freq //= 5 # output isnt persistent (gets overwritten with progress bar)


	assert valloader is not None or 'val_freq' in A.training
	val_freq = A.training.val_freq if 'val_freq' in A.training else None

	# endregion

	# region Stats

	viz_criterion = None
	if 'viz_criterion' in A.training and 'viz_criterion_args' in A.training:
		viz_criterion = util.get_loss_type(A.training.viz_criterion)
		viz_criterion_args = A.training.viz_criterion_args

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

	# endregion

	time_limit = None
	if A.run_mode == 'cluster' and 'time_limit' in A.training:

		print('SOFT TIME LIMIT: {:2.3f} hr'.format(A.training.time_limit)) # given in hours

		start_time = time.time()
		time_limit = A.training.time_limit * 3600 # convert to sec

	assert len(trainloader) >= 1, 'not a single batch in loader'
	records['epoch'] += 1 # incomplete epochs dont count as epochs
	model.pre_epoch('train', records['epoch'])
	loader, epoch_seed = restart_loader(trainloader, epoch_seed, model)

	if inline:
		pbar = tqdm(total=sample_limit, unit='sample', initial=records['train']['total_samples'])

	while (sample_limit is None or records['train']['total_samples'] < sample_limit) \
		and records['total_steps'] < step_limit:

		try:
			start = time.time()
			batch = next(loader)
		except StopIteration:
			model.post_epoch('train', records['epoch'])
			records['stats']['train'].append(train_stats.export())
			loader, epoch_seed = restart_loader(trainloader, epoch_seed, model)
			records['epoch'] += 1
			model.pre_epoch('train', records['epoch'])
			start = time.time()

			batch = next(loader)

		batch = util.to(batch, A.device)

		B = batch.size(0)
		records['total_samples'] += B

		time_stats.update('data', time.time() - start)
		start = time.time()

		out = model.step(batch)
		records['total_steps'] += 1
		if 'loss' in out:
			stats.update('loss', out.loss.detach())

		time_stats.update('model', time.time() - start)
		start = time.time()

		if logger is not None and print_freq is not None and records['total_steps'] % print_freq == 0:

			with torch.no_grad():
				if viz_criterion is not None:
					stats.update('loss-viz', viz_criterion(*[out[n] for n in viz_criterion_args]).detach())

				logger_step(logger, stats, smooths=True)

				if isinstance(model, Visualizable):
					model.visualize(out, logger)

		if not silent:
			loss_info = ' Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
				if stats['loss'].count > 0 else ''
			if inline:
				itr.set_description('{} ep={} (lim={}) {}'.format(mode, epoch, loss_info))
			elif print_freq is None or records['total_steps'] % print_freq == 0:
				print('[ {} ] {} Ep={}/{} Itr={}/{}{}'.format(
					time.strftime("%H:%M:%S"), mode,
					epoch, total_epochs, i + 1, len(loader), loss_info))

				sys.stdout.flush()

		time_stats.update('viz', time.time() - start)

		if val_freq is not None and records['total_steps'] % val_freq == 0:

			if pbar is not None:
				pbar.close()
				pbar = tqdm(total=len(valloader))


			if pbar is not None:
				pbar = tqdm(total=sample_limit, unit='sample', initial=records['train']['total_samples'])






		del out
		torch.cuda.empty_cache()

		records['epoch'] += 1



		model.pre_epoch(mode='train', epoch=records['epoch'], steps=records['total_steps'])
		# try:
		# 	datasets[0].pre_epoch(mode='train', epoch=records['epoch'])
		# except ValueError:
		# 	pass

		train_stats = run_epoch(model, trainloader, A, mode='train', records=records,
		                              logger=logger, silent=False, inline='inline' in A and A.inline)

		model.post_epoch(mode='train', epoch=records['epoch'], steps=records['total_steps'], stats=train_stats)

		records['stats']['train'].append(train_stats.export())

		if valloader is not None:
			model.pre_epoch(mode='val', epoch=records['epoch'], steps=records['total_steps'])

			val_stats = run_epoch(model, valloader, A, mode='val', records=records, unique_tests=A.output.unique_tests,
		                              logger=logger, silent=False, inline='inline' in A and A.inline)

			records['stats']['val'].append(val_stats.export())

			model.post_epoch(mode='val', epoch=records['epoch'], steps=records['total_steps'], stats=val_stats)

		epoch_seed = util.gen_deterministic_seed(epoch_seed)

		if A.output.save_freq > 0 and ((records['epoch']-1) % A.output.save_freq == 0
		                               or records['epoch'] == A.training.epochs):


			_restart_counter += 1

		print()

		if 'RESTART_AFTER' in os.environ and ((_restart_counter+1) % int(os.environ['RESTART_AFTER']) == 0) and (N > i+1):

			print('*** Exiting for restart after {} checkpoints'.format(os.environ['RESTART_AFTER']))
			sys.exit(3)




def basfdrun_continuous(model, loader, A, records, mode='test',
              logger=None, unique_tests=False, silent=False,
              stats=None, inline=False):
	train = mode == 'train'
	if train:
		model.train()
	else:
		model.eval()

	epoch = records['epoch']
	total_epochs = A.training.epochs  # + A.training.start
	print_freq = A.output.print_freq

	viz_criterion = None
	if 'viz_criterion' in A.training and 'viz_criterion_args' in A.training:
		viz_criterion = util.get_loss_type(A.training.viz_criterion)
		viz_criterion_args = A.training.viz_criterion_args

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

	logger_prefix = '{}/{}'.format('{}', mode) if not unique_tests or train else 'zz{}-{}/{}'.format(epoch, '{}',
	                                                                                                 mode)

	total = len(loader)

	max_iter = A.training.max_iter if 'max_iter' in A.training else None
	if max_iter is not None:
		print('\n\nWARNING: not running full epochs\n\n')
		total = max_iter

	itr = enumerate(iter(loader))
	if inline:
		itr = tqdm(itr, total=total, leave=True)

	start = time.time()
	for i, batch in itr:

		if max_iter is not None and i == max_iter:
			print('WARNING: ending epoch prematurely! - hit max_iter')
			del itr
			break

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
					for k, v in stats.smooths().items():
						logger.add('scalar', k, v)

				if isinstance(model, Visualizable):
					model.visualize(out, logger)

		if not silent:
			loss_info = ' Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
				if stats['loss'].count > 0 else ''
			if inline:
				itr.set_description('{} {}/{}{}'.format(mode, epoch, total_epochs, loss_info))
			elif print_freq is None or i % print_freq == 0:
				print('[ {} ] {} Ep={}/{} Itr={}/{}{}'.format(
					time.strftime("%H:%M:%S"), mode,
					epoch, total_epochs, i + 1, len(loader), loss_info))

				sys.stdout.flush()

		time_stats.update('viz', time.time() - start)
		start = time.time()

		del out
		torch.cuda.empty_cache()

	if not silent:
		loss_info = ' Loss: {:.4f} ({:.4f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
			if stats['loss'].count > 0 else ''
		msg = '[ {} ] {} Ep={}/{} complete{}'.format(
			time.strftime("%H:%M:%S"), mode,
			epoch, total_epochs, loss_info)
		border = '-' * len(msg)

		if inline:
			print()
		print(border)
		print(msg)
		print(border)

	if logger is not None:
		logger.flush()
	sys.stdout.flush()

	return stats








