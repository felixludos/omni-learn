
import sys, os, time
import socket
# import traceback, ipdb
from tqdm import tqdm
import yaml
import torch
from .. import util
from ..data.collectors import Info_Dataset
from ..framework import Visualizable, Recordable, Schedulable, Evaluatable
# from .load_data import old_get_loaders as get_loaders # TODO: update

from .model import default_create_model
from .data import default_load_data

from .setup import setup_records, setup_logging
from .loading import load, save_checkpoint
from .data import get_loaders

def iterative_run(A, *args, **kwargs):

	if 'legacy' in A:
		print('\n\n\n\nERROR: Legacy code is no longer supported\n\n\n\n')
		raise Exception('legacy flag set.')

	return new_run_full(A, *args, **kwargs)


def new_run_full(A, get_data=None, get_model=None, get_name=None):
	if get_model is None:
		get_model = default_create_model
	if get_data is None:
		get_data = default_load_data

	if 'device' not in A:
		A.device = 'cuda'

	if A.device == 'cuda' and not torch.cuda.is_available():
		A.device = 'cpu'
		print('cuda not available, falling back to cpu')
	print('Using device: {}'.format(A.device))
	
	if 'cuda' in A.device and not ('no_det' in A and A.no_det):
		torch.backends.cudnn.deterministic = True
		print('Set cudnn backends to deterministic')

	# Set seed
	if 'seed' not in A:
		A.seed = util.gen_random_seed()
	print('Using pegasus seed: {}'.format(A.seed))

	sys.stdout.flush()


	#####################
	# region Create Data/Model
	#####################
	
	if '_loaded' in A and ('soft_load' not in A or not A.soft_load):
		A.loaded = A._loaded
	
	path, extend = None, None
	override = None
	if 'resume' in A:
		path = A.resume
		assert not os.path.isfile(path), 'When resuming you should only specify the directory of the run, not the specific checkpoint'
		if 'extend' in A:
			extend = A.extend
		if 'override' in A:
			A = A.override
			override = A
		else:
			A.clear()
	elif 'loaded' in A:
		path = A.loaded
		A.info.loaded = path

	A, (*datasets, testset), model, ckpt = load(path=path, config=A, mode='train',
	                                            update_config='loaded' not in A or override is not None,
	                                            load_last=True, # load will load the best, rather than last
	                                            load_optim='skip_optim' not in A or not A.skip_optim,
	                                            load_scheduler='skip_scheduler' not in A or not A.skip_scheduler,
	                                              get_model=get_model, get_data=get_data,
	                                              return_args=True, return_ckpt=True, )#strict='load' not in A)

	model.prep(*datasets)

	# endregion

	#####################
	# region Logging
	#####################

	if 'name' not in A:
		A.name = get_name(A) if get_name is not None else None

	if 'save_dir' in A.output: # TODO: is this necessary?
		del A.output.save_dir

	logger = setup_logging(A.output)
	
	# if path is None:
	# 	hparams = model.get_hparams()
	# 	if len(hparams):
	# 		logger.add_hparams(hparams)

	if '_logged_date' in A.output:
		A.info.date = A.output._logged_date

	if ckpt is None: # novel
		records = setup_records(A.training)
		epoch_seed = util.gen_deterministic_seed(A.seed)
	else: # resume, load, complete
		records = ckpt['records']
		epoch_seed = ckpt['epoch_seed']
		if 'loaded' in A: # load
			# del A.load
			print('WARNING: you are loading a previous model!')
			print('Previous model has trained for {} steps, {} epochs'.format(records['total_steps'],
			                                                                  records['epoch']))
			del A.loaded
		else: # resume, complete
			if extend is not None: # resume
				assert 'step_limit' in A.training, 'Cannot extend steps, because there is no limit set'
				A.training.step_limit = extend
				print('Extending training to {} steps'.format(extend))
			print('Running {} more steps'.format(A.training.step_limit - records['total_steps']))

			# if override is not None:
			# 	A.update(override)
			# 	print('Overrode config for resume')

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
						f.write('{:<12} - {} - {} - {}\n'.format(os.environ['JOB_ID'].split('#')[-1],
						                                         socket.gethostname(),
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
		
	num_workers = A.pull('num_workers', 0)
	batch_size = A.dataset.pull('batch_size', 64)
	shuffle = A.dataset.pull('shuffle', True)
	drop_last = A.dataset.pull('drop_last', True)

	# TODO: pull loader info instead of accessing directly.

	loaders = get_loaders(*datasets, batch_size=batch_size, num_workers=num_workers,
	                                            shuffle=shuffle, drop_last=drop_last, silent=True)

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
	# print('Batch size: {} samples'.format(A.dataset.batch_size))

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

	pbar = tqdm if 'inline' in A and A.inline else None


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
	
	eval_only = A.pull('eval_only', False)
	
	if not eval_only and A.training.step_limit - records['total_steps'] > 0:

		print('Training for {} steps'.format(A.training.step_limit - records['total_steps']))
		run_continuous(A, records, model, trainloader, valloader,
		               logger=logger, silent=False, display_samples=False,
	                   epoch_seed=epoch_seed, pbar=pbar)

	else:
		print('No training')


	print('Training complete.')
	
	# endregion
	
	#####################
	# region Evaluation
	#####################
	
	
	
	results = None
	if isinstance(model, Evaluatable):
		
		identifier = A.eval.pull('identifier', 'eval')
		
		print('*'*50)
		print(f'Evaluating trained model: {identifier}')
		print('*'*50)
		
		if 'use_testset' in A.eval and A.eval.use_testset:
			if testset is None:
				testset = get_data(A.dataset, mode='test')
		else:
			print('Test dataset NOT used!')
			testset = None

		if testset is not None:
			testloader = get_loaders(testset, batch_size=A.dataset.batch_size, num_workers=A.num_workers,
			                         shuffle=A.dataset.shuffle, drop_last=A.dataset.drop_last, silent=True)
	
			print('testdata len={}, testloader len={}'.format(len(testset), len(testloader)))
		else:
			testloader = None
		
		if 'use_best' in A.eval and A.eval.use_best and 'save_dir' in A.output and records['checkpoint'] > 0:
			try:
				model, ckpt = load(path=A.output.save_dir, mode='test', get_model=get_model, get_data=None,
				                   return_args=False, return_ckpt=True, force_load_model=True)
				print('Loaded best model, trained for {} iterations'.format(ckpt['records']['total_steps']))
				records = ckpt['records']
			except FileNotFoundError:
				print('Using current model for testing')
		
		records['training_steps'] = records['total_steps']
		
		logger.set_step(records['total_steps'])
		logger.set_tag_format('{}/{}'.format(identifier, '{}'))
		
		info = {
			'_A': A, # full config, probably shouldnt be used
			'A': A.eval, # eval settings
			'datasets': datasets,
			'loaders': loaders,
			
			'identifier': identifier,
			'logger': logger,
			
			'testset': testset,
			'testloader': testloader,
		}
		
		model.eval()
		results = model.evaluate(info)
		
		if results is not None and 'save_dir' in A.output:
			results_path = os.path.join(A.output.save_dir, 'results.pth.tar')
			torch.save(results, results_path)
			print(f'Results saved to {results_path}')
			
	
	# endregion
	
	return A, model, datasets, loaders, records, results


def restart_loader(loader, epoch_seed=None):

	if epoch_seed is not None:
		util.set_seed(epoch_seed)

	loader = iter(loader)

	if epoch_seed is None:
		return loader

	epoch_seed = util.gen_deterministic_seed(epoch_seed)

	return loader, epoch_seed

def gen_logger_prefix(mode, unique_test=False, epoch=None):

	# if unique_test and mode != 'train':
	# 	return 'zz{}-{}/{}'.format(epoch, '{}',mode)

	return '{}/{}'.format('{}', mode)


# def logger_step(logger, stats, smooths=True):
# 	pass




def run_continuous(A, records, model, trainloader, valloader=None,
                   logger=None, silent=False, display_samples=False,
                   stats=None, epoch_seed=None, pbar=None):

	if silent or display_samples:
		raise NotImplementedError

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

		assert save_freq > 100, 'not allowed to save more often than once every 100 steps -- remember 55-8'

		if 'quick_save' not in A.output or not A.output.quick_save:
			A.output.save_freq = len(trainloader)

		print('Will save a checkpoint every {} steps'.format(save_freq))

	print_freq = A.output.print_freq
	if print_freq is not None:
		val_freq = A.training.val_freq
		print('Will log/visualize output every {} steps'.format(print_freq))

	assert valloader is not None or 'val_freq' in A.training
	val_freq = None
	if 'val_freq' in A.training and valloader is not None:
		val_freq = A.training.val_freq
		print('Will evaluate on validation set every {} steps'.format(val_freq))

	print('Each training epoch has {} steps (will run for about {} epochs)'.format(len(trainloader), step_limit // len(trainloader)))

	# endregion

	# region Stats

	viz_criterion = None
	if 'viz_criterion' in A.training and 'viz_criterion_args' in A.training:
		viz_criterion = util.get_loss_type(A.training.viz_criterion)
		viz_criterion_args = A.training.viz_criterion_args

	def gen_stats(stats=None):
		if stats is None:
			stats = util.StatsMeter('loss')
		elif 'loss' not in stats:
			stats.new('loss')
		if viz_criterion is not None and 'loss-viz' not in stats:
			stats.new('loss-viz')
		return stats

	train_stats = gen_stats(stats)
	val_model_stats = None
	if isinstance(model, Recordable):
		train_model_stats = model.stats
		val_model_stats = model.stats.copy()

		train_stats.shallow_join(train_model_stats)

	time_stats = util.StatsMeter('data', 'model', 'viz', 'eval', 'save')
	train_stats.shallow_join(time_stats, fmt='time-{}')

	# endregion

	assert len(trainloader) >= 1, 'not a single batch in loader'
	# records['epoch'] += 1 # incomplete epochs dont count as epochs
	records['checkpoint'] += 1
	model.pre_epoch('train', records['epoch'])
	trainloader.dataset.pre_epoch('train', records['epoch'])
	loader, epoch_seed = restart_loader(trainloader, epoch_seed)

	print('Training dataset len={} loader={}'.format(len(trainloader.dataset), len(trainloader)))

	is_best = False

	keep_going = lambda: (sample_limit is None or records['train']['total_samples'] < sample_limit) \
		and records['total_steps'] < step_limit

	# print('Will save checkpoints every {} steps'.format(save_freq))


	time_limit = None
	if A.run_mode == 'cluster' and 'time_limit' in A.training:

		print('SOFT TIME LIMIT: {:2.2f} hr'.format(A.training.time_limit)) # given in hours

		start_time = time.time()
		time_limit = A.training.time_limit * 3600 # convert to sec

	bar = None if pbar is None else pbar(total=step_limit, initial=records['total_steps'])

	while keep_going():

		# region Training Iteration

		try:
			start = time.time()
			batch = next(loader)

		except StopIteration:
			model.post_epoch('train', records['epoch'], stats=train_stats)
			trainloader.dataset.post_epoch('train', records['epoch'], stats=train_stats)
			records['stats']['train'].append(train_stats.export())
			records['epoch'] += 1
			model.pre_epoch('train', records['epoch'])
			trainloader.dataset.pre_epoch('train', records['epoch'])
			loader, epoch_seed = restart_loader(trainloader, epoch_seed)
			start = time.time()

			batch = next(loader)

		batch = util.to(batch, A.device)

		B = batch.size(0)
		records['total_samples']['train'] += B

		time_stats.update('data', time.time() - start)
		start = time.time()

		out = model.step(batch)
		records['total_steps'] += 1
		if bar is not None:
			bar.update(1)
		if 'loss' in out:
			train_stats.update('loss', out.loss.detach())

		time_stats.update('model', time.time() - start)

		if logger is not None and print_freq is not None and records['total_steps'] % print_freq == 0:

			start = time.time()

			with torch.no_grad():
				if viz_criterion is not None:
					train_stats.update('loss-viz', viz_criterion(*[out[n] for n in viz_criterion_args]).detach())

				logger.set_step(records['total_samples']['train'] if display_samples else records['total_steps'])
				logger.set_tag_format('{}/train')

				with torch.no_grad():
					display = train_stats.smooths() #if smooths else stats.avgs()
					for k, v in display.items():
						logger.add('scalar', k, v)

				if isinstance(model, Visualizable):
					model.visualize(out, logger)

			time_stats.update('viz', time.time() - start)

		del out
		torch.cuda.empty_cache()

		if not silent:
			loss_info = ' Loss: {:.3f} ({:.3f})'.format(train_stats['loss'].val.item(),
			                                            train_stats['loss'].smooth.item()) \
				if train_stats['loss'].count > 0 else ''
			if bar is not None:
				bar.set_description('Train ep={} ckpt={}{}'.format(records['epoch']+1, records['checkpoint'], loss_info))
			elif print_freq is None or (records['total_steps']-1) % print_freq == 0:
				print('[ {} ] Train Ep={} Ckpt={} Itr={}/{}{}'.format(
					time.strftime("%H:%M:%S"), records['epoch']+1, records['checkpoint'],
					records['total_steps'], step_limit, loss_info))

				sys.stdout.flush()

		# endregion

		# region Validation

		if val_freq is not None and records['total_steps'] % val_freq == 0:

			model.eval()
			val_stats = gen_stats()
			if val_model_stats is not None:
				model.stats = val_model_stats
				val_stats.shallow_join(val_model_stats)
			model.pre_epoch('val', records['epoch'])
			valloader.dataset.pre_epoch('val', records['epoch'])

			if not silent:
				if bar is not None:
					bar.close()
					bar = None
					print()
				# print('Evaluating on Validation set')

			vloader = restart_loader(valloader, epoch_seed=None)
			vloader = pbar(enumerate(vloader), total=len(valloader)) if pbar is not None else enumerate(vloader)

			start = time.time()

			for i, batch in vloader:
				if i > 0:
					# break # TESTING
					del out
					torch.cuda.empty_cache()

				batch = util.to(batch, A.device)

				B = batch.size(0)
				records['total_samples']['val'] += B

				out = model.test(batch)
				if 'loss' in out:
					val_stats.update('loss', out.loss.detach())

				if pbar is not None:
					loss_info = ' Loss: {:.3f} ({:.3f})'.format(val_stats['loss'].val.item(), val_stats['loss'].smooth.item()) \
						if val_stats['loss'].count > 0 else ''
					vloader.set_description('Val ep={} ckpt={}{}'.format(records['epoch']+1, records['checkpoint'], loss_info))

			if logger is not None:

				with torch.no_grad():
					if viz_criterion is not None:
						val_stats.update('loss-viz', viz_criterion(*[out[n] for n in viz_criterion_args]).detach())

					logger.set_tag_format('{}/val')

					if isinstance(model, Visualizable):
						model.visualize(out, logger)

					display = val_stats.avgs()  # if smooths else stats.avgs()
					for k, v in display.items():
						logger.add('scalar', k, v)

					logger.set_tag_format('{}/train') # reset

			if pbar is not None:
				vloader.close()
				print()

			del vloader
			del out
			torch.cuda.empty_cache()

			val_loss = val_stats['loss']

			if val_loss.count == 0:
				print('WARNING: loss has not been logged, so the best model cant be tracked')

			best_info = ''
			if 'best' in records and \
					records['best']['loss'] is None or (val_loss.count > 0
					                                    and val_loss.avg <= records['best']['loss']):
				prev = '!' if records['best']['loss'] is None \
					else ', previous was: {:.3f} (ckpt={})'.format(records['best']['loss'],
					                                               records['best']['checkpoint'])
				best_info = ' which is a new best{}'.format(prev)
				records['best']['loss'] = val_loss.avg
				records['best']['checkpoint'] = records['checkpoint']
				is_best = True
			loss_info = ' Loss: {:.3f}{}'.format(val_loss.avg.item(), best_info) \
				if val_loss.count > 0 else ''

			print('[ {} ] Validation Ep={} Ckpt={} Itr={}/{}{}'.format(
				time.strftime("%H:%M:%S"), records['epoch'] + 1, records['checkpoint'],
				records['total_steps'], step_limit, loss_info))

			records['stats']['val'].append(val_stats.export())

			time_stats.update('eval', time.time() - start)
			model.post_epoch('val', records['epoch'], stats=val_stats) # possibly takes scheduler step
			valloader.dataset.post_epoch('val', records['epoch'], stats=val_stats)
			model.train()

			if val_model_stats is not None:
				model.stats = train_model_stats

			# if pbar is not None:
			# 	bar = pbar(total=step_limit, initial=records['total_steps'])

			time_stats.update('eval', time.time() - start)

		# endregion

		# region Save

		if save_freq > 0 and (records['total_steps'] % save_freq == 0 or not keep_going()):
			start = time.time()

			ckpt = {
				'model_str': str(model),
				'model_state': model.state_dict(),

				'records': records,
				'epoch_seed': epoch_seed,
			}

			path = save_checkpoint(ckpt, A.output.save_dir, is_best=is_best, epoch=records['total_steps'])
			best_info = '(new best) ' if is_best else ''
			is_best = False

			time_stats.update('save', time.time() - start)

			if bar is not None:
				bar.close()
				bar = None
				print()

			print('[[ {}checkpoint {} saved to {} ]]'.format(best_info, records['checkpoint'], path))

			if time_limit is not None and ((time.time() - start_time) > time_limit):
				print('*** Exiting for restart after checkpoint {}, time limit ({:2.2f} hr) has been reached'.format(
					records['checkpoint'], A.training.time_limit))
				sys.exit(3) # exit code 3 is special - meaning it the job should be rescheduled on the cluster

			records['checkpoint'] += 1

		if pbar is not None and bar is None:
			bar = pbar(total=step_limit, initial=records['total_steps'])

		# endregion




	# records['epoch'] += 1

	# model.pre_epoch(mode='train', epoch=records['epoch'], steps=records['total_steps'])
	# # try:
	# # 	datasets[0].pre_epoch(mode='train', epoch=records['epoch'])
	# # except ValueError:
	# # 	pass
	#
	# train_stats = run_epoch(model, trainloader, A, mode='train', records=records,
	#                               logger=logger, silent=False, inline='inline' in A and A.inline)
	#
	# model.post_epoch(mode='train', epoch=records['epoch'], steps=records['total_steps'], stats=train_stats)
	#
	# records['stats']['train'].append(train_stats.export())
	#
	# if valloader is not None:
	# 	model.pre_epoch(mode='val', epoch=records['epoch'], steps=records['total_steps'])
	#
	# 	val_stats = run_epoch(model, valloader, A, mode='val', records=records, unique_tests=A.output.unique_tests,
	#                               logger=logger, silent=False, inline='inline' in A and A.inline)
	#
	# 	records['stats']['val'].append(val_stats.export())
	#
	# 	model.post_epoch(mode='val', epoch=records['epoch'], steps=records['total_steps'], stats=val_stats)
	#
	# epoch_seed = util.gen_deterministic_seed(epoch_seed)
	#
	# if A.output.save_freq > 0 and ((records['epoch']-1) % A.output.save_freq == 0
	#                                or records['epoch'] == A.training.epochs):
	#
	#
	# 	_restart_counter += 1
	#
	# print()
	#
	# if 'RESTART_AFTER' in os.environ and ((_restart_counter+1) % int(os.environ['RESTART_AFTER']) == 0) and (N > i+1):
	#
	# 	print('*** Exiting for restart after {} checkpoints'.format(os.environ['RESTART_AFTER']))
	# 	sys.exit(3)



#
# def basfdrun_continuous(model, loader, A, records, mode='test',
#               logger=None, unique_tests=False, silent=False,
#               stats=None, inline=False):
# 	train = mode == 'train'
# 	if train:
# 		model.train()
# 	else:
# 		model.eval()
#
# 	epoch = records['epoch']
# 	total_epochs = A.training.epochs  # + A.training.start
# 	print_freq = A.output.print_freq
#
# 	viz_criterion = None
# 	if 'viz_criterion' in A.training and 'viz_criterion_args' in A.training:
# 		viz_criterion = util.get_loss_type(A.training.viz_criterion)
# 		viz_criterion_args = A.training.viz_criterion_args
#
# 	if stats is None:
# 		stats = util.StatsMeter('loss')
# 		if isinstance(model, Recordable):
# 			stats.shallow_join(model.stats)
# 	elif 'loss' not in stats:
# 		stats.new('loss')
# 	if viz_criterion is not None and 'loss-viz' not in stats:
# 		stats.new('loss-viz')
#
# 	time_stats = util.StatsMeter('data', 'model', 'viz')
# 	stats.shallow_join(time_stats, fmt='time-{}')
#
# 	logger_prefix = '{}/{}'.format('{}', mode) if not unique_tests or train else 'zz{}-{}/{}'.format(epoch, '{}',
# 	                                                                                                 mode)
#
# 	total = len(loader)
#
# 	max_iter = A.training.max_iter if 'max_iter' in A.training else None
# 	if max_iter is not None:
# 		print('\n\nWARNING: not running full epochs\n\n')
# 		total = max_iter
#
# 	itr = enumerate(iter(loader))
# 	if inline:
# 		itr = tqdm(itr, total=total, leave=True)
#
# 	start = time.time()
# 	for i, batch in itr:
#
# 		if max_iter is not None and i == max_iter:
# 			print('WARNING: ending epoch prematurely! - hit max_iter')
# 			del itr
# 			break
#
# 		batch = util.to(batch, A.device)
#
# 		B = batch.size(0)
#
# 		records['total_samples'][mode] += B
#
# 		time_stats.update('data', time.time() - start)
# 		start = time.time()
#
# 		if train:
# 			out = model.step(batch)
# 		else:
# 			out = model.test(batch)
# 		if 'loss' in out:
# 			stats.update('loss', out.loss.detach())
#
# 		time_stats.update('model', time.time() - start)
# 		start = time.time()
#
# 		if logger is not None and print_freq is not None and i % print_freq == 0:
#
# 			logger.set_step(records['total_samples'][mode])
# 			logger.set_tag_format(logger_prefix)
#
# 			with torch.no_grad():
# 				if viz_criterion is not None:
# 					stats.update('loss-viz', viz_criterion(*[out[n] for n in viz_criterion_args]).detach())
#
# 				if logger is not None:
# 					for k, v in stats.smooths().items():
# 						logger.add('scalar', k, v)
#
# 				if isinstance(model, Visualizable):
# 					model.visualize(out, logger)
#
# 		if not silent:
# 			loss_info = ' Loss: {:.3f} ({:.3f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
# 				if stats['loss'].count > 0 else ''
# 			if inline:
# 				itr.set_description('{} {}/{}{}'.format(mode, epoch, total_epochs, loss_info))
# 			elif print_freq is None or i % print_freq == 0:
# 				print('[ {} ] {} Ep={}/{} Itr={}/{}{}'.format(
# 					time.strftime("%H:%M:%S"), mode,
# 					epoch, total_epochs, i + 1, len(loader), loss_info))
#
# 				sys.stdout.flush()
#
# 		time_stats.update('viz', time.time() - start)
# 		start = time.time()
#
# 		del out
# 		torch.cuda.empty_cache()
#
# 	if not silent:
# 		loss_info = ' Loss: {:.4f} ({:.4f})'.format(stats['loss'].val.item(), stats['loss'].smooth.item()) \
# 			if stats['loss'].count > 0 else ''
# 		msg = '[ {} ] {} Ep={}/{} complete{}'.format(
# 			time.strftime("%H:%M:%S"), mode,
# 			epoch, total_epochs, loss_info)
# 		border = '-' * len(msg)
#
# 		if inline:
# 			print()
# 		print(border)
# 		print(msg)
# 		print(border)
#
# 	if logger is not None:
# 		logger.flush()
# 	sys.stdout.flush()
#
# 	return stats








