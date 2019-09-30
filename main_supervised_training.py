# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

import torch.nn.functional as F

# Global imports
import os
import sys
import shutil
import time
import numpy as np
import torch.multiprocessing as mp
import h5py as hf

import foundation as fd
import foundation.util as util
import foundation.nets as nets
import foundation.data as data
import pydub
import torchaudio
import options

from loading import load_classify_model, save_model

def collect_datafiles(dirs, cond=None):
	if cond is None:
		cond = lambda x: True
		
	files = []
	for d in dirs:
		files.extend(util.crawl(d, cond))
		
	return files


######################
def main(): #
	########################
	## Parse args
	global args, num_train_iter
	parser = options.setup_system_options()
	args = parser.parse_args()
	
	now = time.strftime("%y-%m-%d-%H%M%S")
	if args.log_date:
		args.name += '_' + now
	args.save_dir = os.path.join(args.save_root, args.name)
	# args.save_dir = os.path.join(args.save_root, args.name)
	print('Save dir: {}'.format(args.save_dir))
	if args.tblog or args.txtlog:
		util.create_dir(args.save_dir)
		print('Logging in {}'.format(args.save_dir))
	logger = util.Logger(args.save_dir, tensorboard=args.tblog, txt=args.txtlog)
	
	args.device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
	print('Using {}'.format(args.device))
	
	# Set seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	########################
	# Load data
	########################
	
	args.loss_weight = None
	
	if args.dataset == 'musicnet':
		pass # TODO
	elif args.dataset == 'yt':
		raise NotImplementedError
	elif args.dataset == 'archive':
		raise NotImplementedError
	else:
		raise Exception('unknown data type: {}'.format(args.dataset))
	
	print('Found {} samples in {}'.format(len(args.data_files), args.data))
	
	print('traindata len={}, trainloader len={}'.format(len(traindata), len(trainloader)))
	print('valdata len={}, valloader len={}'.format(len(valdata), len(valloader)))
	
	print('Batch {} samples, each {} ms long with stepsize {}'.format(args.batch_size, args.seq_len, args.step_len))
	
	########################
	# Load model/optim
	########################
	
	args.curr_lr = args.lr

	model = load_classify_model(args=args)
	
	optimizer = nets.get_optimizer(args.optimization, model.parameters(), lr=args.lr,
								   momentum=args.momentum, weight_decay=args.weight_decay)
	
	criterion = nets.get_loss_type(args.loss_type, weight=args.loss_weight)
	
	model.to(args.device)
	optimizer.to(args.device)
	
	assert not args.resume, 'resuming not supported yet'

	if args.no_test:
		print('Will not run test data after training/validation')

	########################
	# Train/Val
	args.sample_ctr = {'train': 0, 'val': 0, 'test': 0}
	best_val_loss, best_epoch = float('inf'), 0
	for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
		# Adjust learning rate
		args.curr_lr = nets.adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay, args.decay_epochs)

		# Train for one epoch
		train_stats = iterate(trainloader, model, logger,
							  mode='train', optimizer=optimizer, criterion=criterion,
							  epoch=epoch)

		# Evaluate on validation set
		val_stats = iterate(valloader, model, logger,
							mode='val', criterion=criterion, epoch=epoch)

		# Find best loss
		val_loss = val_stats['loss']  # all models have a flow-vel error - only from predicted/computed delta
		is_best = (val_loss.avg < best_val_loss)
		prev_best_loss = best_val_loss
		prev_best_epoch = best_epoch
		if is_best:
			best_val_loss = val_loss.avg
			best_epoch = epoch + 1
			print('==== Epoch: {}, Improved on previous best loss ({:.5f}) from epoch {}. Current: {:.5f} ===='.format(
				epoch + 1, prev_best_loss, prev_best_epoch, val_loss.avg))
		else:
			print('==== Epoch: {}, Did not improve on best loss ({:.5f}) from epoch {}. Current: {:.5f} ===='.format(
				epoch + 1, prev_best_loss, prev_best_epoch, val_loss.avg))

		model.to('cpu')
		optimizer.to('cpu')

		path = save_model({
			'epoch': epoch,
			'args': args,
			
			'best_loss': best_val_loss,
			'best_epoch': best_epoch,
			
			'train_stats': train_stats,
			'val_stats': val_stats,
			
			'train_data': traindata,
			'val_data': valdata,
			'test_data': testdata,
			
			'model_state': model.state_dict(),
			'optim_state': optimizer.state_dict(),
		}, save_dir=args.save_dir, is_best=is_best, epoch=epoch+1)
		
		model.to(args.device)
		optimizer.to(args.device)
		
		if path is not None:
			print('--- checkpoint saved: {} ---'.format(path))
			
		print()

	if args.no_test:
		print('Training complete - and no test')
		return 0
	
	assert False, 'testing not ready yet'

	# Load best model for testing (not latest one)
	print("=> loading best model from '{}'".format(args.save_dir + "/model_best.pth.tar"))
	checkpoint = torch.load(args.save_dir + "/model_best.pth.tar")
	num_train_iter = checkpoint['train_iter']
	model.load_state_dict(checkpoint['model_state_dict'])
	print("=> loaded best checkpoint (epoch {}, train iter {})".format(checkpoint['epoch'], num_train_iter))
	best_epoch = checkpoint['best_epoch']
	print('==== Best validation loss: {} was from epoch: {} ===='.format(checkpoint['best_loss'],
																		 best_epoch))

	# Do final testing (if not asked to evaluate)
	# (don't create the data loader unless needed, creates 4 extra threads)
	print('==== Evaluating trained network on test data ====')
	test_stats = iterate(testloader, model, tblogger, -1, mode='test', criterion=criterion, epoch=args.epochs)
	print('==== Best validation loss: {} was from epoch: {} ===='.format(best_val_loss,
																		 best_epoch))

	# Save final test error
	save_checkpoint({
		'args': args,
		'test_stats': test_stats
	}, False, savedir=args.save_dir, filename='test_stats.pth.tar')

	# Close log file
	logfile.close()

###############
### Main iterate function (train/test/val)
def iterate(loader, model, logger, mode='test', optimizer=None, criterion=None, epoch=0):

	print('=================== Mode: {}, Epoch: {} ==================='.format(mode, epoch))

	# Setup avg time & stats:
	time_stats = util.StatsMeter('data', 'fwd', 'bwd', 'viz')
	stats = util.StatsMeter('loss', 'viz', )

	# Switch model modes
	train = (mode == 'train')
	if train:
		assert (optimizer is not None), "Please pass in an optimizer if we are iterating in training mode"
		model.train()
	else:
		assert (mode == 'test' or mode == 'val'), "Mode can be train/test/val. Input: {}".format(mode)
		model.eval()

	viz_criterion = nets.get_loss_type('mse', weight=args.loss_weight)
	if criterion is None:
		criterion = viz_criterion
	
	start = time.time()
	for itr, sample in enumerate(loader):
		# ============ Load data ============#
		
		# TODO
		
		args.sample_ctr[mode] += sample.size(0) # batch size TODO: probably fix
		
		# Measure data loading time
		time_stats.update('data', time.time() - start)

		# ============ FWD pass + Compute loss ============#
		# Start timer
		start = time.time()

		loss = 0
		
		# TODO

		# Save loss
		stats.update('loss', loss.detach())
		loss *= args.loss_scale

		# Measure FWD time
		time_stats.update('fwd', time.time() - start)

		# ============ Gradient backpass + Optimizer step ============#
		# Compute gradient and do optimizer update step (if in training mode)
		if (train):
			# Start timer
			start = time.time()

			# Backward pass & optimize
			optimizer.zero_grad()  # Zero gradients
			loss.backward()  # Compute gradients - BWD pass
			optimizer.step()  # Run update step

			# Measure BWD time
			time_stats.update('bwd', time.time() - start)

		# ============ Visualization ============#
		# Start timer
		start = time.time()
		
		# no more grads
		with torch.no_grad():
			
			# TODO: use viz_criterion
			viz_loss = 0
			
			stats.update('viz', viz_loss)

			if itr % args.disp_freq == 0:
				### Print statistics
				print_stats(mode, epoch=epoch, curr=itr, total=len(loader), stats=stats, times=time_stats)

				### logging
				info = stats.vals(mode + '-{}')
				info.update(time_stats.vals(mode + '-timing-{}'))
				if train:
					info['train-lr'] = args.curr_lr  # Plot current learning rate
				
				logger.update(info, step=args.sample_ctr[mode])
				
			# Measure viz time
			time_stats.update('viz', time.time() - start)

		start = time.time() # start timer for data loading next iteration

	### Print stats at the end
	print('========== Mode: {}, Epoch: {}, Final results =========='.format(mode, epoch))
	print_stats(mode, epoch=epoch, curr=itr+1, total=total_itr, stats=stats)
	print('========================================================')

	# Return the loss & flow loss
	return stats

################
### Print statistics
def print_stats(mode, epoch, curr, total, stats, times):
	# Print loss
	print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Samples: {}, '
		  'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
		mode, epoch+1, args.start_epoch + args.epochs, args.sample_ctr[mode],
		curr+1, total, loss=stats['loss']))

################ RUN MAIN
if __name__ == '__main__':
	mp.set_start_method('spawn')
	main()
