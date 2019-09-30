# Torch imports
import torch
import torchvision
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
	shuffle = [True, False]
	
	if args.dataset == 'musicnet':
		
		assert False
		
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
		
		split_info = torch.load(args.split_path)
		args.record_split = split_info['paths']
		
		args.record_split = list(args.record_split[0]) + list(args.record_split[1]), args.record_split[2]
		
		
		assert len(args.records) == sum(map(len, args.record_split)), 'bad record split'
		
		yt_dataset = datasets.Yt_Dataset #datasets.Full_Song_Yt_Dataset if args.full_records else datasets.Yt_Dataset
		
		#extra = {'batch_size':args.batch_size} if args.single_step else {}
		#if args.single_step:
		#	shuffle = [False, False, False]
		
		traindata, testdata = [yt_dataset(recs, seq_len=args.seq_len, hop=args.hop, lbl_name=lbl_name,
		                                  begin=None, end=None,) for recs in args.record_split]
		
	elif args.dataset == 'archive':
		raise NotImplementedError

	elif args.dataset == 'mnist':

		traindata = MNIST('/home/fleeb/workspace/ml_datasets', train=True, transform=transforms.ToTensor(), download=True)
		testdata = MNIST('/home/fleeb/workspace/ml_datasets', train=False, transform=transforms.ToTensor(), download=True)

		#traindata, valdata = fd.data.split_dataset(traindata, trainper=1-args.val_per)
		
	else:
		raise Exception('unknown data type: {}'.format(args.dataset))
	
	args.in_shape = 1, args.sample_rate * args.seq_len // 1000
	
	if args.data_type == 'spec':

		traindata, testdata = [datasets.MEL_Dataset(dataset, hop=args.mel_hop,
													ws=args.mel_ws, n_mels=args.mel_n)
										for dataset in [traindata, testdata]]
		
		args.in_shape = traindata.output
		
		print('Using MEL {} as input'.format(args.in_shape))
	else:
		print('Using wave {} as input'.format(args.in_shape))
	
	
	trainloader, testloader = [DataLoader(dataset, batch_size=args.batch_size,
													 shuffle=sh, num_workers=args.num_workers)
										  for dataset, sh in zip([traindata, testdata], shuffle)]
	
	print('Found {} records in {}'.format(sum([len(recs) for recs in args.record_split]), args.data))
	print('Split: train={}, test={}'.format(*[len(recs) for recs in args.record_split]))
	
	print('traindata len={}, trainloader len={}'.format(len(traindata), len(trainloader)))
	#print('valdata len={}, valloader len={}'.format(len(valdata), len(valloader)))
	
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
	
	j, g = model
	
	j_optim, g_optim = nets.get_optimizer(args.optimization, j.parameters(), lr=args.lr,
								   momentum=args.momentum, weight_decay=args.weight_decay), \
					nets.get_optimizer(args.optimization, g.parameters(), lr=args.lr,
					                   momentum=args.momentum, weight_decay=args.weight_decay)
	
	optimizer = j_optim, g_optim
	
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
		args.curr_lr = nets.adjust_learning_rate(optimizer[0], lr=args.lr, epoch=epoch, decay_rate=args.lr_decay, decay_epochs=args.decay_epochs)
		args.curr_lr = nets.adjust_learning_rate(optimizer[1], lr=args.lr, epoch=epoch, decay_rate=args.lr_decay,
		                                         decay_epochs=args.decay_epochs)
		
		# Train for one epoch
		train_stats = iterate(trainloader, model, logger,
							  mode='train', optimizer=optimizer, epoch=epoch)

		# Evaluate on validation set
		# val_stats = iterate(valloader, model, logger,
		# 					mode='val', cls_criterion=criterion, epoch=epoch)
		#
		if 'loss' in train_stats:
			print('*** EPOCH {}/{} pretend: train={:.4f} ***'.format(epoch+1, args.epochs+args.start_epoch, train_stats['loss'].avg))

		# Find best loss
		val_loss = train_stats['loss'].avg  # all models have a flow-vel error - only from predicted/computed delta
		is_best = (val_loss < best_val_loss)
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
			#'val_stats': val_stats,
			
			'train_data': traindata,
			#'val_data': valdata,
			'test_data': testdata,
			
			'model_state': model.state_dict(),
			'j_optim_state': optimizer[0].state_dict(),
			'g_optim_state': optimizer[1].state_dict(),
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
def iterate(loader, model, logger, mode='test', optimizer=None, epoch=0):

	print('=================== Mode: {}, Epoch: {} ==================='.format(mode, epoch))

	# Setup avg time & stats:
	time_stats = util.StatsMeter('data', 'fwd', 'bwd', 'viz')
	stats = util.StatsMeter('loss', 'fake', 'real', 'fake-max', 'real-min', 'pretend', 'rms-data', 'rms-gen', 'real-max', 'gen-max')

	# Switch model modes
	train = (mode == 'train')
	if train:
		assert (optimizer is not None), "Please pass in an optimizer if we are iterating in training mode"
		model.train()
	else:
		assert (mode == 'test' or mode == 'val'), "Mode can be train/test/val. Input: {}".format(mode)
		model.eval()

	judge, gen = model
	
	if optimizer is not None:
		j_optim, g_optim = optimizer
	
	start = time.time()
	for itr, sample in enumerate(loader):
		# ============ Load data ============#
		
		x,y = sample
		
		x = x.float().to(args.device)
		if args.data_type != 'spec':
			x = x.unsqueeze(1)
		y = y.long().to(args.device)

		# if args.dataset == 'mnist':
		# 	x = x.view(x.size(0), -1)
		
		args.sample_ctr[mode] += x.size(0) # batch size
		
		# Measure data loading time
		time_stats.update('data', time.time() - start)

		# ============ FWD pass + Compute loss ============#
		# Start timer
		start = time.time()

		loss = 0
		
		cls = None
		if args.gen_use_cls:
			cls = util.to_one_hot(y, args.out_dim)
		
		# print(x.size(), y.size())
		#
		# print(gen.spec(x).size())
		
		audio = gen(num=x.size(0), cls=cls)
		
		# print(gen.mel.size(), gen.phase.size(), audio.size())
		#
		# import matplotlib.pyplot as plt
		#
		# plt.figure()
		# plt.plot(x[0,0].cpu().numpy())
		# plt.title('real')
		# plt.tight_layout()
		#
		# plt.figure()
		# plt.plot(audio[0,0].detach().cpu().numpy())
		# plt.title('gen')
		# plt.tight_layout()
		#
		# plt.figure()
		# plt.imshow(gen.mel[0].detach().cpu().numpy())
		# plt.title('mel')
		# plt.tight_layout()
		#
		# plt.figure()
		# plt.imshow(gen.phase[0].detach().cpu().numpy())
		# plt.title('phase')
		# plt.tight_layout()
		#
		# plt.show()
		
		# quit()
		
		if args.model == 'judge-mel':
			mel, phase = audio
			genmel, genphase = mel, phase
			
			# print(mel.max(), mel.min(), phase.max(), phase.min())
			# print(mel.size(), phase.size())
			
			audio = torch.cat([mel, torch.cos(phase), torch.sin(phase)], -1)
			
			# print(audio.size())
			
			mel, phase = gen.spec(x, ret_phase=True)
			mel, phase = mel.permute(0, 2, 1), phase.permute(0, 2, 1)
			
			xmel, xphase = mel, phase
			
			stats.update('real-max', mel.detach().abs().max())
			
			# print(mel.max(), mel.min(), phase.max(), phase.min())
			# print(mel.size(), phase.size())
			
			info = torch.cat([mel, torch.cos(phase), torch.sin(phase)], -1)
			
			# print(info.size())
			
			real = judge(info, cls=cls)
			fake = judge(audio.detach(), cls=cls)
			
			# print(real.size(), fake.size())
			
			# import matplotlib.pyplot as plt
			#
			# plt.figure()
			# plt.imshow(genmel[0].detach().cpu().numpy())
			# plt.title('mel')
			# plt.tight_layout()
			#
			# plt.figure()
			# plt.imshow(genphase[0].detach().cpu().numpy())
			# plt.title('phase')
			# plt.tight_layout()
			#
			# plt.figure()
			# plt.imshow(mel[0].detach().cpu().numpy())
			# plt.title('true mel')
			# plt.tight_layout()
			#
			# plt.figure()
			# plt.imshow(phase[0].detach().cpu().numpy())
			# plt.title('true phase')
			# plt.tight_layout()
			#
			# plt.show()
			
			
			#quit()
			
		
		else:
			
			real = judge(x, cls=cls)
			fake = judge(audio.detach(), cls=cls)
		
		
		
		fwd_time = time.time() - start
		bwd_time = 0
		
		if train:
			
			start = time.time()
			
			j_optim.zero_grad()
			(-real).mean().backward()
			fake.mean().backward()
			j_optim.step()
			
			bwd_time += time.time() - start
			
		if args.judge_clip > 0:
			for param in judge.parameters():
				param.data.clamp_(-args.judge_clip, args.judge_clip)
		
		if itr % args.judge_steps == 0:
		
			for _ in range(args.gen_steps):
				
				start = time.time()
				
				audio = gen(num=x.size(0), cls=cls)
				
				if args.model == 'judge-mel':
					mel, phase = audio
					
					audio = torch.cat([mel, torch.cos(phase), torch.sin(phase)], -1)
				
				pretend = judge(audio, cls=cls)
				
				fwd_time += time.time() + start
		
				if train:
					# Start timer
					start = time.time()
		
					g_optim.zero_grad()  # Zero gradients
					(-pretend).mean().backward()  # Compute gradients - BWD pass
					g_optim.step()  # Run update step
					
					bwd_time += time.time() - start
		
		time_stats.update('fwd', time.time() - start)
		if train:
			time_stats.update('bwd', time.time() - start)

		# ============ Visualization ============#
		# Start timer
		start = time.time()
		
		# no more grads
		with torch.no_grad():
			
			rms_data = x.pow(2).mean(-1).sqrt().mean()
			stats.update('rms-data', rms_data)
			
			rms_gen = audio.pow(2).mean(-1).sqrt().mean()
			stats.update('rms-gen', rms_gen)
			
			# 'fake', 'real', 'pretend'
			
			stats.update('gen-max', audio.abs().max().detach())
			
			stats.update('loss', torch.abs(rms_data-rms_gen))
			stats.update('fake', fake.detach().mean())
			stats.update('fake-max', fake.detach().max())
			stats.update('real', real.detach().mean())
			stats.update('real-min', real.detach().min())
			stats.update('pretend', pretend.detach().mean())
			
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
				
				
				if itr % (args.disp_freq*5) == 0:
					
					N = 8
					
					try:
					
						gms = genmel[:N].detach()
						rms = xmel[:N].detach()
						
						gps = genphase[:N].detach()
						rps = xphase[:N].detach()
						
						# print(gms.size(), rms.size(), gps.size(), rps.size())
						
						B, H, W = gms.size()
						
						ms = torch.stack([gms, rms], 1).view(-1, 1, H, W)  # .expand(2*B, 3, H, W)
						ps = torch.stack([gps, rps], 1).view(-1, 1, H, W)  # .expand(2 * B, 3, H, W)
						
						# print(ms.size(), ps.size())
						
						mgrid = torchvision.utils.make_grid(ms, nrow=2, normalize=True).unsqueeze(0)
						pgrid = torchvision.utils.make_grid(ps, nrow=2, normalize=True).unsqueeze(0)
						
						# print(mgrid.size(), pgrid.size())
						
						logger.update_images({mode + '-mel': mgrid.detach().cpu().numpy(),
						                      mode + '-phase': pgrid.detach().cpu().numpy(), },
						                     step=args.sample_ctr[mode])
					
					
					except:
						pass
					
					
					# ms = []
					# ps = []
					# for gm, rm, gp, rp in zip(gms, rms, grps, rps):
					#
					# 	ms.append(gm)
					# 	ms.append(ms)
					
					
					
					
				
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
