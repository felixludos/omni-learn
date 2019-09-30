# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

import torch.nn.functional as F

# Global imports
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import h5py as hf

import foundation as fd
import foundation.util as util
from foundation import train
from foundation import nets

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
	parser = train.setup_unsup_options()
	args = parser.parse_args()
	
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
	
	# Set seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	try:
		torch.cuda.manual_seed(args.seed)
	except:
		pass

	########################
	# Load data
	########################

	datasets = train.load_data(args=args)
	shuffles = [True, False, False]

	loaders = [DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers) for d, s in zip(datasets, shuffles)]

	trainloader, testloader = loaders[0], loaders[-1]
	valloader = None if len(loaders) == 2 else loaders[1]

	print('traindata len={}, trainloader len={}'.format(len(datasets[0]), len(trainloader)))
	if valloader is not None:
		print('valdata len={}, valloader len={}'.format(len(datasets[1]), len(valloader)))
	print('testdata len={}, testloader len={}'.format(len(datasets[-1]), len(testloader)))
	print('Batch size: {} samples'.format(args.batch_size))
	
	########################
	# Load model/optim
	########################

	model = train.load_unsup_model(args=args)

	print(model)

	print('Model has {} parameters'.format(util.count_parameters(model)))
	
	optimizer = nets.get_optimizer(args.optim_type, model.parameters(), lr=args.lr,
								   momentum=args.momentum, weight_decay=args.weight_decay)
	
	model.to(args.device)
	#optimizer.to(args.device)
	
	assert not args.resume, 'resuming not supported yet'

	scheduler = None
	if args.decay_epochs > 0 and args.decay_factor > 0:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
													step_size=args.decay_epochs,
													gamma=args.decay_factor)


	########################
	# Train/Val
	########################

	if args.no_test:
		print('Will not run test data after training')

	args.total_samples = {'train': 0, 'val': 0, 'test': 0}
	best_val_loss, best_epoch = float('inf'), 0
	for epoch in range(args.start_epoch, args.start_epoch+args.epochs):

		if scheduler is not None:
			# Adjust learning rate
			scheduler.step()

		train_stats = util.StatsMeter('lr', tau=0.1)
		train_stats.update('lr', optimizer.param_groups[0]['lr'])

		# Train for one epoch
		train_stats = train.run_unsup_epoch(loader=trainloader, model=model, logger=logger, args=args,
							optim=optimizer, unique_tests=args.unique_tests, print_freq=args.print_freq,
							mode='train', epoch=epoch, stats=train_stats)
		
		val_stats = util.StatsMeter(tau=0.1)

		# Evaluate on validation set
		if valloader is not None:
			val_stats = train.run_unsup_epoch(loader=valloader, model=model, logger=logger, args=args,
								optim=None, unique_tests=args.unique_tests, print_freq=args.print_freq,
								mode='val', epoch=epoch, stats=val_stats)
		else:
			val_stats = None

		# Find best loss
		val_loss = val_stats['loss'] if val_stats is not None else train_stats['loss']  # all models have a flow-vel error - only from predicted/computed delta
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
		#optimizer.to('cpu')

		path = train.save_checkpoint({
			'epoch': epoch,
			'args': args,
			
			'best_loss': best_val_loss,
			'best_epoch': best_epoch,
			
			'train_stats': train_stats,
			'val_stats': val_stats,
			
			# 'traindata': datasets[0],
			# 'valdata': datasets[1] if len(datasets)>2 else None,
			# 'testdata': datasets[-1],
			
			'model_state': model.state_dict(),
			'optim_state': optimizer.state_dict(),
		}, save_dir=args.save_dir, is_best=is_best, epoch=epoch+1)
		
		model.to(args.device)
		#optimizer.to(args.device)
		
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




################ RUN MAIN
if __name__ == '__main__':
	#mp.set_start_method('spawn')
	main()
