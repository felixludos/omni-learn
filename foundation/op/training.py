
import sys, os
from tqdm import tqdm
import time

import torch

import omnifig as fig

from ..framework import Recordable, Schedulable, Evaluatable, Visualizable

from .. import util

# from .loading import load_config, load_records, setup_logging, setup_records, \
# 	wrap_datasets, wrap_transaction, save_checkpoint
from .loading import respect_config
from .model import load_model
from .data import load_data
from .evaluation import eval_model


@fig.Script('train', description='Train new/existing models')
def iterative_training(A=None, run=None):
	'''
	This is the entry for the training script for new or existing runs.
	Existing runs (or models) can be specified using "path", "load",
	or "resume" with the run name
	'''
	#####################
	# region Loading
	#####################

	respect_config(A)

	if run is None:
		assert A is not None, 'either run or A must not be None'
		A.push('run._type', 'run', overwrite=False)
		run = A.pull('run')
	
	A = run.get_config()
	
	datasets = run.get_datasets()
	model = run.get_model()
	
	logger = run.get_logger()
	
	run.prepare()
	
	loaders = run.get_loaders()
	
	# endregion
	#######################
	# region Smart Defaults
	#######################
	
	if 'train' not in datasets:
		raise Exception(f'No training dataset found (how did this happen?)')
		
	for key in ['train', 'val', 'test']:
		if key in datasets:
			print(f'{key}data len={len(datasets[key])}, {key}loader len={len(loaders[key])}')
	
	trainloader = loaders['train']
	epoch_len = len(trainloader)
	
	tau = A.push('training.stats.tau', max(0.01, min(100/epoch_len, 0.1)), overwrite=False)
	util.set_default_tau(tau)
	if isinstance(model, Recordable):
		model.stats.set_tau(tau)
	A.push('output.print_freq', min(max(20, epoch_len // 40), 200), overwrite=False)
	
	epochs = A.pull('training.epochs', 10)
	step_limit = A.push('training.step_limit', epochs * epoch_len, overwrite=False)
	
	no_val = A.pull('training.no_val', False)
	A.push('training.val_freq', None if no_val else epoch_len, overwrite=False)

	inline = A.pull('inline', False)
	pbar = tqdm if inline else None
	
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
	# region Run Training
	#####################
	
	remaining_steps = step_limit - run.get_total_steps()
	
	if remaining_steps > 0:
		print(f'Training for {remaining_steps} steps')
		# run.prepare(model, trainloader)
		run.continuous(pbar=pbar)
		print('Training complete.')
	
	else:
		print('No training')
	
	# endregion
	#####################
	# region Run Evaluation
	#####################
	
	if isinstance(model, Evaluatable):
		eval_model(A, run=run)
		
	# endregion
	
	return run


# def restart_loader(loader, records=None):
#
# 	epoch_seed, skip_batches = None, None
# 	if records is not None:
# 		epoch_seed = records.get('epoch_seed', None)
# 		skip_batches = records.get('batch', 0)
#
# 	if epoch_seed is not None:
# 		util.set_seed(epoch_seed)
#
# 	assert skip_batches is None or len(loader) > skip_batches, f'Trying to skip too much: ' \
# 	                                                           f'{skip_batches} vs {len(loader)}'
#
# 	loader = iter(loader)
#
# 	if skip_batches is not None and skip_batches > 0:
# 		try:
# 			loader.skip(skip_batches)
# 		except AttributeError:
# 			print('WARNING: no auto skip implemented')
# 			for _ in range(skip_batches):
# 				next(loader)
#
# 	if epoch_seed is None:
# 		return loader
#
# 	records['epoch_seed'] = util.gen_deterministic_seed(epoch_seed)
#
# 	return loader



# def run_continuous(run, pbar=None):
#
# 	A = run.get_config()
#
# 	model = run.get_config()
#
# 	(*trainloaders, _) = run.get_loaders()
# 	if len(trainloaders) > 1:
# 		trainloader, valloader = trainloaders
# 	else:
# 		trainloader = trainloaders[0]
# 		valloader = None
#
# 	stats = None
#
# 	# region Limits/Freqs
#
# 	step_limit = A.pull('training.step_limit', None)
# 	sample_limit = A.pull('training.sample_limit', None)
#
# 	save_dir = A.pull('output.save_dir', None)
# 	save_freq = A.pull('output.save_freq', -1)
# 	if save_dir is None:
# 		print('WARNING: No save_dir provided')
# 		save_freq = None
# 	else:
# 		A.export(os.path.join(save_dir, 'config.yaml'))
# 	if save_freq is not None and 0 < save_freq < len(trainloader):
# 		print('WARNING: saving more than once per epoch: checkpoint every {} iterations'.format(save_freq))
#
# 		assert save_freq > 100, 'not allowed to save more often than once every 100 steps -- remember 55-8'
#
# 		quick_save = A.pull('output.quick_save', False) # force saving so frequently
#
# 		if not quick_save:
# 			save_freq = len(trainloader)
# 	if save_freq is not None and save_freq > 0:
# 		print(f'Will save a checkpoint every {save_freq} steps')
#
# 	silent = A.pull('output.silent', False)
# 	display_samples = A.pull('output.display_samples', False) # count in terms of samples instead of iterations
# 	display_smoothed = A.pull('output.display_smoothed', True)
#
# 	print_freq = A.pull('output.print_freq', None)
# 	unique_tests = A.pull('output.unique_tests', False)
# 	val_freq = A.pull('training.val_freq', None)
# 	if print_freq is not None:
# 		print('Will log/visualize output every {} steps'.format(print_freq))
# 	if val_freq is not None and valloader is None:
# 		print('No validation set provided, so val freq set to 0')
# 		val_freq = None
#
# 	print('Each training epoch has {} steps (will run for about {} epochs)'.format(len(trainloader), step_limit // len(trainloader)))
#
# 	# endregion
#
# 	# region Stats
#
# 	device = A.pull('device', 'cpu')
#
# 	viz_criterion = A.pull('training.viz_criterion', None)
# 	viz_criterion_args = A.pull('training.viz_criterion_args', [])
# 	if viz_criterion is not None and len(viz_criterion_args) == 0:
# 		print('WARNING: no args provided for viz_criterion')
# 		viz_criterion = None
#
# 	def gen_stats(stats=None):
# 		if stats is None:
# 			stats = util.StatsMeter('loss')
# 		elif 'loss' not in stats:
# 			stats.new('loss')
# 		if viz_criterion is not None and 'loss-viz' not in stats:
# 			stats.new('loss-viz')
# 		return stats
#
# 	train_stats = gen_stats(stats)
# 	val_model_stats = None
# 	if isinstance(model, Recordable):
# 		train_model_stats = model.stats
# 		val_model_stats = model.stats.copy()
#
# 		train_stats.shallow_join(train_model_stats)
#
# 	time_stats = util.StatsMeter('data', 'model', 'viz', 'eval', 'save')
# 	train_stats.shallow_join(time_stats, fmt='time-{}')
#
# 	# endregion
#
# 	is_best = False
#
# 	skip_prev_batches = A.pull('training.skip_prev_batches', False)
# 	if not skip_prev_batches:
# 		records['batch'] = 0
#
# 	# TODO: unique tests in logging
#
# 	assert len(trainloader) >= 1, 'not a single batch in loader'
# 	model.pre_epoch('train', records)
# 	trainloader.dataset.pre_epoch('train', records)
# 	loader = restart_loader(trainloader, records)
#
# 	print(f'Training dataset len={len(trainloader.dataset)} loader={len(trainloader)}')
#
# 	keep_going = lambda: (sample_limit is None or records['total_samples']['train'] < sample_limit) \
# 		and records['total_steps'] < step_limit
#
# 	if run is not None:
# 		run.starting(records)
#
# 	time_limit = A.pull('training.time_limit', None)
# 	if time_limit is not None:
# 		print(f'SOFT TIME LIMIT: {time_limit:2.2f} hrs')
# 		start_time = time.time()
# 		time_limit = 3600 * time_limit
#
# 	bar = None if pbar is None else pbar(total=step_limit, initial=records['total_steps'])
#
# 	while keep_going():
#
# 		# region Training Iteration
#
# 		try:
# 			start = time.time()
# 			batch = next(loader)
# 			records['batch'] += 1
#
# 		except StopIteration:
# 			model.post_epoch('train', records, stats=train_stats)
# 			trainloader.dataset.post_epoch('train', records, stats=train_stats)
# 			records['stats']['train'].append(train_stats.export())
# 			records['epoch'] += 1
# 			records['batch'] = 1
# 			model.pre_epoch('train', records)
# 			trainloader.dataset.pre_epoch('train', records)
# 			loader = restart_loader(trainloader, records)
# 			start = time.time()
#
# 			batch = next(loader)
#
# 		batch = util.to(batch, device)
#
# 		B = batch.size(0)
# 		records['total_samples']['train'] += B
#
# 		time_stats.update('data', time.time() - start)
# 		start = time.time()
#
# 		out = model.step(batch)
# 		records['total_steps'] += 1
# 		if bar is not None:
# 			bar.update(1)
# 		if 'loss' in out:
# 			train_stats.update('loss', out.loss.detach())
#
# 		time_stats.update('model', time.time() - start)
#
# 		if logger is not None and print_freq is not None and records['total_steps'] % print_freq == 0:
#
# 			start = time.time()
#
# 			with torch.no_grad():
# 				if viz_criterion is not None:
# 					train_stats.update('loss-viz', viz_criterion(*[out[n] for n in viz_criterion_args]).detach())
#
# 				logger.set_step(records['total_samples']['train'] if display_samples else records['total_steps'])
# 				logger.set_tag_format('{}/train')
#
# 				with torch.no_grad():
# 					display = train_stats.smooths() if display_smoothed else stats.avgs()
# 					for k, v in display.items():
# 						logger.add('scalar', k, v)
#
# 				if isinstance(model, Visualizable):
# 					model.visualize(out, logger)
#
# 			time_stats.update('viz', time.time() - start)
#
# 		del out
# 		torch.cuda.empty_cache()
#
# 		if not silent:
# 			loss_info = ' Loss: {:.3f} ({:.3f})'.format(train_stats['loss'].val.item(),
# 			                                            train_stats['loss'].smooth.item()) \
# 				if train_stats['loss'].count > 0 else ''
# 			if bar is not None:
# 				bar.set_description('Train ep={} ckpt={}{}'.format(records['epoch']+1, records['checkpoint'], loss_info))
# 			elif print_freq is None or (records['total_steps']-1) % print_freq == 0:
# 				print('[ {} ] Train Ep={} Ckpt={} Itr={}/{}{}'.format(
# 					time.strftime("%H:%M:%S"), records['epoch']+1, records['checkpoint'],
# 					records['total_steps'], step_limit, loss_info))
#
# 				sys.stdout.flush()
#
# 		# endregion
#
# 		# region Validation
#
# 		if val_freq is not None and (records['total_steps'] % val_freq == 0 or not keep_going()):
#
# 			if run is not None:
# 				run.validating(records)
#
# 			model.eval()
# 			val_stats = gen_stats()
# 			if val_model_stats is not None:
# 				model.stats = val_model_stats
# 				val_stats.shallow_join(val_model_stats)
# 			model.pre_epoch('val', records)
# 			valloader.dataset.pre_epoch('val', records)
#
# 			if not silent:
# 				if bar is not None:
# 					bar.close()
# 					bar = None
# 					print()
# 				# print('Evaluating on Validation set')
#
# 			vloader = restart_loader(valloader)
# 			vloader = pbar(enumerate(vloader), total=len(valloader)) \
# 				if pbar is not None else enumerate(vloader)
#
# 			start = time.time()
#
# 			for i, batch in vloader:
#
# 				batch = util.to(batch, device)
#
# 				B = batch.size(0)
# 				records['total_samples']['val'] += B
#
# 				out = model.test(batch)
# 				if 'loss' in out:
# 					val_stats.update('loss', out.loss.detach())
#
# 				if pbar is not None:
# 					loss_info = ' Loss: {:.3f} ({:.3f})'.format(val_stats['loss'].val.item(), val_stats['loss'].smooth.item()) \
# 						if val_stats['loss'].count > 0 else ''
# 					vloader.set_description('Val ep={} ckpt={}{}'.format(records['epoch']+1, records['checkpoint'], loss_info))
#
# 			records['num_validations'] += 1
# 			records['validation'] = records['total_steps']
#
# 			if logger is not None:
#
# 				with torch.no_grad():
# 					if viz_criterion is not None:
# 						val_stats.update('loss-viz', viz_criterion(*[out[n] for n in viz_criterion_args]).detach())
#
# 					logger_id = '{}/val{}'.format('{}', records['num_validations']) if unique_tests else '{}/val'
# 					logger.set_tag_format(logger_id)
#
# 					if isinstance(model, Visualizable):
# 						model.visualize(out, logger)
#
# 					display = val_stats.avgs()  # if smooths else stats.avgs()
# 					for k, v in display.items():
# 						logger.add('scalar', k, v)
#
# 					logger.set_tag_format('{}/train') # reset
#
# 			if pbar is not None:
# 				vloader.close()
# 				print()
#
# 			del vloader
# 			del out
# 			torch.cuda.empty_cache()
#
# 			val_loss = val_stats['loss']
#
# 			if val_loss.count == 0:
# 				print('WARNING: loss has not been logged, so the best model cant be tracked')
#
# 			best_info = ''
# 			if 'best' in records and \
# 					records['best']['loss'] is None or (val_loss.count > 0
# 					                                    and val_loss.avg <= records['best']['loss']):
# 				prev = '!' if records['best']['loss'] is None \
# 					else ', previous was: {:.3f} (ckpt={})'.format(records['best']['loss'],
# 					                                               records['best']['checkpoint'])
# 				best_info = f' which is a new best{prev}'
# 				records['best']['loss'] = val_loss.avg.item()
# 				records['best']['checkpoint'] = records['checkpoint']
# 				is_best = True
# 			loss_info = ' Loss: {:.3f}{}'.format(val_loss.avg.item(), best_info) \
# 				if val_loss.count > 0 else ''
#
# 			print('[ {} ] Validation Ep={} Ckpt={} Itr={}/{}{}'.format(
# 				time.strftime("%H:%M:%S"), records['epoch'] + 1, records['checkpoint'],
# 				records['total_steps'], step_limit, loss_info))
#
# 			records['stats']['val'].append(val_stats.export())
#
# 			time_stats.update('eval', time.time() - start)
# 			model.post_epoch('val', records, stats=val_stats) # possibly takes scheduler step
# 			valloader.dataset.post_epoch('val', records, stats=val_stats)
# 			model.train()
#
# 			if val_model_stats is not None:
# 				model.stats = train_model_stats
#
# 			# if pbar is not None:
# 			# 	bar = pbar(total=step_limit, initial=records['total_steps'])
#
# 			if run is not None:
# 				run.validated(records, val_stats)
#
# 		# endregion
#
# 		# region Save
#
# 		if save_freq > 0 and (records['total_steps'] % save_freq == 0 or not keep_going()):
#
# 			start = time.time()
#
# 			records['checkpoint'] = records['total_steps']
#
# 			save_checkpoint(save_dir, model, records,
# 			                steps=records['checkpoint'], is_best=is_best)
#
# 			best_info = '(new best) ' if is_best else ''
# 			is_best = False
#
# 			time_stats.update('save', time.time() - start)
#
# 			if bar is not None:
# 				bar.close()
# 				bar = None
# 				print()
#
# 			print('[[ {}checkpoint {} saved to {} ]]'.format(best_info, records['checkpoint'], save_dir))
#
# 			if run is not None:
# 				run.checkpointing(records)
#
# 			if time_limit is not None and ((time.time() - start_time) > time_limit):
# 				print('*** Exiting for restart after checkpoint {}, time limit ({:2.2f} hr) has been reached'.format(
# 					records['checkpoint'], time_limit))
# 				sys.exit(3) # exit code 3 is special - meaning it the job should be rescheduled on the cluster
#
# 		if pbar is not None and bar is None:
# 			bar = pbar(total=step_limit, initial=records['total_steps'])
#
# 		# endregion




