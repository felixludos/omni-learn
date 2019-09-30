# Torch imports
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision import  transforms
from torchvision.datasets import MNIST
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
from foundation import nets
import foundation.data as data
import pydub
import torchaudio
import options

from loading import load_model, save_model
import datasets

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
	parser = options.setup_classify_options()
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
	
	args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
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
	shuffle = [True, False, False]
	
	if args.dataset == 'musicnet':
		
		assert len(args.data) == 1 and os.path.basename(args.data[0]) == 'musicnet.h5', 'Not the right data: {}'.format(args.data)
		
		path = args.data[0]
		
		split = torch.load(args.split_path)
		
		args.record_split = split['ids']
		
		traindata, valdata, testdata = [datasets.MusicNet_Dataset(path, ids=ids, seq_len=args.seq_len, hop=args.hop,
														 begin=None, end=None) for ids in split['ids']]
		
		args.loss_weight = torch.from_numpy(np.bincount(split['cats'][0], weights=split['lens'][0])).float()
		args.loss_weight = args.loss_weight.sum() / args.loss_weight
		
		
		args.out_dim = 4
		
	elif args.dataset == 'yt':
		
		args.records = []
		
		for d in args.data:
			args.records.extend([os.path.join(d, f) for f in util.crawl(d, cond=lambda x: '.h5' in x)])
			
		args.records = np.array(args.records)
		
		lbl_name = 'gid' if args.data_mode == 'genre' else 'mid'
		args.out_dim = 14 if args.data_mode == 'genre' else 10
		print('Predicting {}'.format(args.data_mode))
		
		try:
			split_info = torch.load(args.split_path)
			args.record_split = split_info['paths']
			args.loss_weight = split_info['loss-weights']
			
			assert len(args.records) == sum(map(len,args.record_split)), 'bad record split'
			
		except Exception as e:
			
			print('Creating new splits: {}'.format(e))
		
			lens = []
			lbls = []
			for rec in args.records:
				with hf.File(rec, 'r') as f:
					lens.append(f['wav'].shape[0])
					lbls.append(f.attrs[lbl_name])
			
			lens = np.array(lens)
			lbls = np.array(lbls)
			
			split_failed = 1
			while split_failed > 0:
				
				orders = util.split_vals(np.arange(len(lens)), [args.train_per, args.val_per, 1-args.train_per-args.val_per])
				
				nlbl = [lbls[o] for o in orders]
				
				counts = np.hstack([np.bincount(l, minlength=args.out_dim) for l in nlbl])
				
				if (counts==0).any():
					
					print()
				
					if split_failed > 10:
						raise Exception('Splitting failed too many times')
					
					if split_failed:
						print('WARNING: Data split failed, retrying...')
						
					split_failed += 1
					
				else:
					
					nlens = [lens[o] for o in orders]
					
					args.loss_weight = nlens[0].sum() / np.bincount(nlbl[0], nlens[0], minlength=args.out_dim)
					args.loss_weight = torch.from_numpy(args.loss_weight).float()
					
					d = np.bincount(lbls, lens, minlength=args.out_dim) / lens.sum()
					d = '[' + ', '.join(['{:.3f}'.format(x) for x in d]) + ']'
					
					print('Overall: {}'.format(d))
					
					names = ['Train', 'Val', 'Test']
					for b, l, n in zip(nlbl, nlens, names):
						d = np.bincount(b, l, minlength=args.out_dim) / l.sum()
						d = '[' + ', '.join(['{:.3f}'.format(x) for x in d]) + ']'
						print('{}: {}'.format(n, d))
						
					break
		
			args.record_split = [args.records[o] for o in orders]
	
			nlens = [lens[o] for o in orders]
			nlbls = [lbls[o] for o in orders]
	
			# torch.save({'paths':args.record_split, 'loss-weights':args.loss_weight,
			# 			'lens': nlens, 'lbls':nlbls}, '/home/fleeb/workspace/ml_datasets/audio/yt/split.pth.tar')
	
		# print(len(args.records), [len(a) for a in args.record_split])
		# quit()
		
		yt_dataset = datasets.Yt_Dataset #datasets.Full_Song_Yt_Dataset if args.full_records else datasets.Yt_Dataset
		
		#extra = {'batch_size':args.batch_size} if args.single_step else {}
		#if args.single_step:
		#	shuffle = [False, False, False]
		
		traindata, valdata, testdata = [yt_dataset(recs, seq_len=args.seq_len, hop=args.hop, begin=None, end=None,) for recs in args.record_split]
		
	elif args.dataset == 'archive':
		raise NotImplementedError

	elif args.dataset == 'mnist':

		traindata = MNIST('/home/fleeb/workspace/ml_datasets', train=True, transform=transforms.ToTensor(), download=True)
		testdata = MNIST('/home/fleeb/workspace/ml_datasets', train=False, transform=transforms.ToTensor(), download=True)

		traindata, valdata = fd.data.split_dataset(traindata, trainper=1-args.val_per)
		
	else:
		raise Exception('unknown data type: {}'.format(args.dataset))
	
	args.in_shape = 1, args.sample_rate * args.seq_len // 1000
	
	if args.data_type == 'spec':

		traindata, valdata, testdata = [datasets.MEL_Dataset(dataset, hop=args.mel_hop,
													ws=args.mel_ws, n_mels=args.mel_n)
										for dataset in [traindata, valdata, testdata]]
		
		args.in_shape = traindata.output
		
		print('Using MEL {} as input'.format(args.in_shape))
	else:
		print('Using wave {} as input'.format(args.in_shape))
	
	
	trainloader, valloader, testloader = [DataLoader(dataset, batch_size=args.batch_size,
													 shuffle=sh, num_workers=args.num_workers)
										  for dataset, sh in zip([traindata, valdata, testdata], shuffle)]
	
	print('Found {} records in {}'.format(sum([len(recs) for recs in args.record_split]), args.data))
	print('Split: train={}, val={}, test={}'.format(*[len(recs) for recs in args.record_split]))
	
	print('traindata len={}, trainloader len={}'.format(len(traindata), len(trainloader)))
	print('valdata len={}, valloader len={}'.format(len(valdata), len(valloader)))
	
	print('Using {} workers to load data'.format(args.num_workers))
	
	#print('Loss weights: {}'.format('['+', '.join(['{:.3f}'.format(w) for w in args.loss_weight.numpy()]) + ']'))
	
	print('Batch {} samples, each {} ms long with stepsize {}'.format(args.batch_size, args.seq_len, args.step_len))
	
	########################
	# Load model/optim
	########################
	
	args.curr_lr = args.lr
	
	if args.fc_dims[0] == 0:
		args.fc_dims = None
	if args.rec_dim == 0:
		args.rec_dim = None

	model = load_model(args=args)
	print(model)

	optimizer = nets.get_optimizer(args.optimization, model.parameters(), lr=args.lr,
								   momentum=args.momentum, weight_decay=args.weight_decay)
	
	args.loss_type = 'nll' if model.normalized else 'cross-entropy'
	
	print('Using {} loss since model is {}normalized'.format(args.loss_type, '' if model.normalized else 'not '))
	
	criterion = nets.get_loss_type(args.loss_type, weight=args.loss_weight).to(args.device)
	
	model.to(args.device)
	
	assert not args.resume, 'resuming not supported yet'

	if args.no_test:
		print('Will not run test data after training/validation')

	########################
	# Train/Val
	args.sample_ctr = {'train': 0, 'val': 0, 'test': 0}
	best_val_loss, best_epoch = float('inf'), 0
	for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
		# Adjust learning rate
		args.curr_lr = nets.adjust_learning_rate(optimizer, lr=args.lr, epoch=epoch, decay_rate=args.lr_decay, decay_epochs=args.decay_epochs)

		# Train for one epoch
		train_stats = iterate(trainloader, model, logger,
							  mode='train', optimizer=optimizer, cls_criterion=criterion,
							  epoch=epoch)

		# Evaluate on validation set
		val_stats = iterate(valloader, model, logger,
							mode='val', cls_criterion=criterion, epoch=epoch)
		
		if 'accuracy' in val_stats:
			print('*** EPOCH {}/{} accuracy: train={:.4f} ({:.3f}), val={:.4f} ({:.3f}) ***'.format(epoch+1, args.epochs+args.start_epoch, train_stats['accuracy'].avg, train_stats['loss'].avg, val_stats['accuracy'].avg, val_stats['loss'].avg))

		# Find best loss
		val_loss = val_stats['accuracy'].avg  # all models have a flow-vel error - only from predicted/computed delta
		is_best = (val_loss > best_val_loss)
		prev_best_loss = best_val_loss
		prev_best_epoch = best_epoch
		if is_best:
			best_val_loss = val_loss
			best_epoch = epoch + 1
			print('==== Epoch: {}, Improved on previous best loss ({:.5f}) from epoch {}. Current: {:.5f} ===='.format(
				epoch + 1, prev_best_loss, prev_best_epoch, val_loss))
		else:
			print('==== Epoch: {}, Did not improve on best loss ({:.5f}) from epoch {}. Current: {:.5f} ===='.format(
				epoch + 1, prev_best_loss, prev_best_epoch, val_loss))

		model.to('cpu')

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
	test_stats = iterate(testloader, model, tblogger, -1, mode='test', cls_criterion=criterion, epoch=args.epochs)
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
def iterate(loader, model, logger, mode='test', optimizer=None, cls_criterion=None, epoch=0):

	print('=================== Mode: {}, Epoch: {} ==================='.format(mode, epoch))

	# Setup avg time & stats:
	time_stats = util.StatsMeter('data', 'fwd', 'bwd', 'viz')
	stats = util.StatsMeter('loss', 'accuracy', 'confidence', 'min', 'loss-rec', 'loss-cls', 'loss-var',
							'q-mu-std', 'q-lsigma-mean', 'q-mu-max', 'q-lsigma-min')

	# Switch model modes
	train = (mode == 'train')
	if train:
		assert (optimizer is not None), "Please pass in an optimizer if we are iterating in training mode"
		model.train()
	else:
		assert (mode == 'test' or mode == 'val'), "Mode can be train/test/val. Input: {}".format(mode)
		model.eval()

	#viz_criterion = nets.get_loss_type('nll') if model.normalized else nets.get_loss_type('cross-entropy')
	
	rec_criterion = nets.get_loss_type(args.rec_loss)
	
	assert args.rec_loss_wt + args.cls_loss_wt + args.var_loss_wt > 0, 'must have some loss'

	# print(model)
	# print(optimizer)
	# print(criterion)
	#
	# ds = MNIST('/home/fleeb/workspace/ml_datasets', train=True, transform=transforms.ToTensor(), download=True)
	#
	# loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	
	hidden_db = None
	if args.full_records:
		
		ty = 2 if args.rec_type == 'lstm' else 1
		
		hidden_db = torch.zeros(loader.dataset.num, ty, args.rec_num_layers, model.rec_dim).float().to(args.device)
		
	start = time.time()
	for itr, sample in enumerate(loader):
		# ============ Load data ============#
		
		if args.full_records:
			i,x,y = sample
			assert len(set(i.numpy())) == len(i)
			hidden = hidden_db[i].clone().permute(1,2,0,3)#.contiguous()
			
			if hidden.size(0) == 1:
				hidden = hidden.squeeze(0)
			
		else:
			x,y = sample
			hidden = None
		
		x = x.float().to(args.device)
		if args.data_type != 'spec':
			x = x.unsqueeze(1)
		y = y.long().to(args.device)

		if args.dataset == 'mnist':
			x = x.view(x.size(0), -1)
		
		args.sample_ctr[mode] += x.size(0) # batch size
		
		# Measure data loading time
		time_stats.update('data', time.time() - start)

		# ============ FWD pass + Compute loss ============#
		# Start timer
		start = time.time()

		loss = 0
		
		q = model(x, decode=False)
		
		#print(q.size())
			
		cls, latent = q[..., :args.out_dim], q[..., args.out_dim:]
		
		D = latent.size(-1) // 2
		mu, log_sigma = latent[..., :D], latent[..., D:].clamp(min=-3)
		log_var = 2 * log_sigma
		
		stats.update('q-mu-std', mu.std().detach())
		stats.update('q-mu-max', mu.abs().max().detach())
		stats.update('q-lsigma-mean', log_sigma.mean().detach())
		stats.update('q-lsigma-min', log_sigma.min().detach())
		
		distrib = Normal(loc=mu, scale=log_sigma.exp())
		
		if args.noisy_rec:
			probs = F.softmax(cls, -1)
			q = torch.cat([distrib.rsample(), probs], -1)
			
			#print('noisy')
		
		# pred = model.decode(q)
		# print(pred.size(), x.size())
		# print('err', x.size(-1) - pred.size(-1))
		# quit()
		
		#print(cls.size(), latent.size(), q.size())
		
		if args.dense_labeling:
			cls = cls.permute(0, 2, 1)
			y = y.unsqueeze(-1).expand(-1, cls.size(-1))
		else:
			cls = cls[:, -1].contiguous()
		
		# print(cls.size(), y.size())
		
		### Cls Loss
		
		cls_loss = cls_criterion(cls, y)
		
		# print('cls',cls_loss)
		
		stats.update('loss-cls', cls_loss.detach())
			
		loss += args.cls_loss_wt * cls_loss
		
		### Rec Loss
		
		pred = model.decode(q)
		
		# print(pred.size(), x.size())
		
		rec_loss = rec_criterion(pred, x)
		
		stats.update('loss-rec', rec_loss.detach())
		
		# print('rec', rec_loss)
		
		loss += args.rec_loss_wt * rec_loss
		
		### Var Loss
		
		#print(mu.size(), log_sigma.size())
		
		var_loss = (mu.pow(2) - log_var + log_var.exp() - 1).mean() / 2
		
		stats.update('loss-var', var_loss.detach())
		
		loss += args.var_loss_wt * var_loss
			
		#quit()

		# Save loss
		stats.update('loss', loss.detach())
		loss *= args.loss_scale

		# Measure FWD time
		time_stats.update('fwd', time.time() - start)

		# ============ Gradient backpass + Optimizer step ============#
		# Compute gradient and do optimizer update step (if in training mode)
		if train:
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
			
			vpred = cls.clone().detach()
			
			if args.dense_labeling:
				vpred = vpred[:, :, -1]#.contiguous()
				y = y[:,0]
			
			conf, pick = vpred.max(-1)
			
			stats.update('min', vpred.min().detach())
			
			confidence = conf.detach()
			correct = pick.sub(y).eq(0).float().detach()
			
			stats.update('confidence', confidence.mean())
			stats.update('accuracy', correct.mean())
			
			# TODO: use viz_criterion
			#viz_loss = viz_criterion(vpred, y).detach()
			
			#stats.update('viz', viz_loss)

			if itr % args.disp_freq == 0:
				### Print statistics
				print_stats(mode, epoch=epoch, curr=itr, total=len(loader), stats=stats, times=time_stats)

				#pos = loader.dataset._current / 44100
				#print(pos.mean(), pos.std(), pos.max(), pos.min())
				#print(hidden_db.view(hidden_db.size(0), -1).sum(-1).eq(0).float().mean().item())

				### logging
				info = stats.vals(mode + '-{}')
				info.update(time_stats.vals(mode + '-timing-{}'))
				if train:
					info['train-lr'] = args.curr_lr  # Plot current learning rate
				
				logger.update(info, step=args.sample_ctr[mode])
				
			# Measure viz time
			time_stats.update('viz', time.time() - start)

		start = time.time() # start timer for data loading next iteration

	# Return the loss & flow loss
	return stats

################
### Print statistics
def print_stats(mode, epoch, curr, total, stats, times):
	# Print loss
	print('Mode: {}, Epoch: [{}/{}], Iter: [{}/{}], Samples: {}, '
		  'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
		mode, epoch+1, args.start_epoch + args.epochs, curr+1, total, args.sample_ctr[mode],
		 loss=stats['loss']))

################ RUN MAIN
if __name__ == '__main__':
	#mp.set_start_method('spawn')
	main()
