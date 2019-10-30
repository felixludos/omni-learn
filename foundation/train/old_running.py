

import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torchvision
from ..framework import Generative, Recordable
from .. import util
from .. import models
from .load_model import save_checkpoint


#####################
# old


def run_rl_training(gen, agent, args=None,
                    logger=None, stats=None, tau=0.1,
                    save_freq=None, print_freq=1,
                    num_iter=None, continue_gen_stats=False):
	save_dir = None
	if hasattr(args, 'save_dir'):
		save_dir = args.save_dir

	if stats is None:
		stats = util.StatsMeter(tau=tau)
	stats.new('time-gen', 'time-learn', 'time-viz')

	gen = iter(gen)

	if hasattr(agent, 'stats'):
		stats.shallow_join(agent.stats)
	if hasattr(gen, 'stats'):
		stats.shallow_join(gen.stats)
	if hasattr(args, 'tau'):
		stats.set_tau(args.tau)

	agent.eval()

	start = time.time()

	for itr, rollouts in enumerate(gen):

		stats.update('time-gen', time.time() - start)
		########################
		# Train Agent
		start = time.time()

		agent.train()
		agent.learn(**rollouts)

		stats.update('time-learn', time.time() - start)
		########################
		# Visualize/Log stats
		start = time.time()

		N_steps, N_episodes = (gen.gen.steps_generated(), gen.gen.episodes_generated()) \
			if continue_gen_stats else (gen.steps_generated(), gen.episodes_generated())

		if logger is not None:
			info = stats.vals()
			info['performance'] = stats['rewards'].smooth.item()
			logger.update(info, step=N_steps)

		if print_freq is not None and itr % print_freq == 0:
			print("[ {} ] {:}/{:} (ep={}) : last={:5.3f} max={:5.3f} - {:5.3f} ".format(
				(time.strftime("%m-%d-%y %H:%M:%S")),
				N_steps, args.budget_steps,
				N_episodes,
				stats['rewards'].val.item(),
				stats['rewards'].max.item(),
				stats['rewards'].smooth.item(),
				))

		if save_freq is not None and itr % save_freq == 0 and save_dir is not None:
			path = save_checkpoint({
				'agent_state_dict': agent.state_dict(),
				'stats': stats,
				'args': args,
				'steps': N_steps,
				'episodes': N_episodes,
			}, args.save_dir, epoch=itr)
			print('--- checkpoint saved at: {} ---'.format(path))

		stats.update('time-viz', time.time() - start)
		start = time.time()

		agent.eval()

		if num_iter is not None and num_iter <= itr:
			break

	return stats


def run_unsup_epoch(model, loader, args, mode='test', optim=None,
                    epoch=None, print_freq=-1, logger=None, unique_tests=False, silent=False,
                    stats_callback=None, stats=None, viz_loss=True):
	train = mode == 'train'
	if train:
		model.train()
	else:
		model.eval()
	if not hasattr(args, 'total_samples'):
		args.total_samples = {'train': 0, 'test': 0}

	viz_criterion = nn.MSELoss() if viz_loss else None

	print_freq = max(1, len(loader) // 100) if print_freq < 0 else print_freq
	if print_freq == 0:
		print_freq = None

	if stats is None:
		stats = util.StatsMeter('loss', tau=max(min(1 / len(loader), 0.1), 0.01))
	elif 'loss' not in stats:
		stats.new('loss')

	if viz_loss and 'viz-loss' not in stats:
		stats.new('viz-loss')

	time_stats = util.StatsMeter('data', 'fwd', 'bwd', 'viz')
	stats.shallow_join(time_stats, prefix='time-')

	logger_prefix = '{}-{}'.format(mode, '{}') if not unique_tests or train else '{}{}-{}'.format(
		mode, epoch + 1 + args.start_epoch, '{}')

	itr = iter(loader)
	start = time.time()
	for i, sample in enumerate(itr):

		sample = sample.to(args.device)

		args.total_samples[mode] += sample.size(0)

		time_stats.update('data', time.time() - start)
		start = time.time()

		out = model.get_loss(sample, stats=stats)
		loss = out['loss']
		stats.update('loss', loss.detach())

		loss *= args.loss_scale

		time_stats.update('fwd', time.time() - start)
		start = time.time()

		if train:
			optim.zero_grad()
			loss.backward()
			optim.step()

		time_stats.update('bwd', time.time() - start)
		start = time.time()

		if stats_callback is not None:
			assert False
			stats_callback(stats, model, sample)

		if print_freq is not None and i % print_freq == 0:
			if logger is not None:

				B, C, H, W = x.size()

				if 'original' in out and 'reconstruction' in out:
					stats.update('viz-loss', viz_criterion(rec, x).detach())

				vals = stats.smooths(logger_prefix)

				logger.update(vals, args.total_samples[mode])

				if i % print_freq * 10 == 0:  # update images

					if 'original' in out and 'reconstruction' in out:
						N = min(4, B)

						viz_x, viz_rec = out['original'][:N], out['reconstruction'][:N]

						imgs = torch.stack([viz_x, viz_rec], 0).view(-1, C, H, W)
						imgs = torchvision.utils.make_grid(imgs, nrow=N).permute(1, 2, 0).unsqueeze(0)

						info = {logger_prefix.format('rec'): imgs.cpu().numpy()}

					try:
						gen = model.generate(N=N).detach()
						imgs = torchvision.utils.make_grid(gen, nrow=N // 2).permute(1, 2, 0).unsqueeze(0)

						info[logger_prefix.format('gen')] = imgs.cpu().numpy()

					except AttributeError:
						pass

					logger.update_images(info, args.total_samples[mode])

			if not silent:
				print('[ {} ] {} Ep={}/{} Itr={}/{} Loss: {:.3f} ({:.3f})'.format(
					time.strftime("%H:%M:%S"), mode,
					epoch + args.start_epoch + 1, args.epochs, i + 1, len(loader),
					stats['loss'].val, stats['loss'].smooth))

		time_stats.update('viz', time.time() - start)
		start = time.time()

	if not silent:
		msg = '[ {} ] {} Ep={}/{} complete Loss: {:.4f} ({:.4f})'.format(
			time.strftime("%H:%M:%S"), mode,
			epoch + args.start_epoch + 1, args.epochs,
			stats['loss'].val, stats['loss'].smooth)
		border = '-' * 50
		print(border)
		print(msg)
		print(border)

	return stats


def run_transition_epoch(model, loader, args, mode='test', optim=None, criterion=None,
                         epoch=None, print_freq=-1, logger=None, unique_tests=False, silent=False,
                         stats_callback=None, stats=None, viz_loss=True):
	train = mode == 'train'
	if train:
		model.train()
	else:
		model.eval()
	if not hasattr(args, 'total_samples'):
		args.total_samples = {'train': 0, 'test': 0}

	viz_criterion = nn.MSELoss() if viz_loss else None
	if criterion is None:
		criterion = nn.MSELoss()

	print_freq = max(1, len(loader) // 100) if print_freq < 0 else print_freq
	if print_freq == 0:
		print_freq = None

	if stats is None:
		stats = util.StatsMeter('loss', tau=max(min(1 / len(loader), 0.1), 0.01))
	elif 'loss' not in stats:
		stats.new('loss-total', 'loss-pred', 'loss-model')

	if viz_loss and 'viz-loss' not in stats:
		stats.new('loss-viz')

	time_stats = util.StatsMeter('data', 'fwd', 'bwd', 'viz')
	stats.shallow_join(time_stats, prefix='time-')

	logger_prefix = '{}-{}'.format(mode, '{}') if not unique_tests or train else '{}{}-{}'.format(
		mode, epoch + 1 + args.start_epoch, '{}')

	itr = iter(loader)
	start = time.time()
	for i, sample in enumerate(itr):

		states = sample['states'].to(args.device)
		controls = sample['controls'].to(args.device)

		states = states.permute(1, 0, 2)
		controls = controls.permute(1, 0, 2)

		q = None
		if 'context' in sample:
			context = sample['context'].to(args.device)
			q = context.permute(1, 0, 2)

		assert states.size(0) == controls.size(0) + 1, '{} and {}'.format(states.shape, controls.shape)

		args.total_samples[mode] += controls.size(0)

		x, nx = states[0], states[1:]
		U = controls  # K, B, C

		time_stats.update('data', time.time() - start)
		start = time.time()

		px = model.sequence(x, U, q)

		model_loss = model.get_loss(q)
		stats.udpate('loss-model', model_loss.detach())

		if hasattr(args, 'pred_len'):
			nx = nx[-args.pred_len:]
			px = px[-args.pred_len:]
		pred_loss = criterion(px, nx)
		stats.update('loss-pred', pred_loss.detach())

		loss = model_loss + pred_loss
		stats.update('loss-total', loss)
		loss *= args.loss_scale

		time_stats.update('fwd', time.time() - start)
		start = time.time()

		if train:
			optim.zero_grad()
			loss.backward()
			optim.step()

		time_stats.update('bwd', time.time() - start)
		start = time.time()

		if stats_callback is not None:
			assert False
			stats_callback(stats, model, sample)

		if print_freq is not None and i % print_freq == 0:
			if logger is not None:
				stats.update('loss-viz', viz_criterion(px.detach(), nx))

				vals = stats.smooths(logger_prefix)

				logger.update(vals, args.total_samples[mode])

			# if i % print_freq * 10 == 0:  # update images
			# 	N = min(4, B)
			#
			# 	viz_x, viz_rec = x[:N], rec[:N]
			#
			# 	imgs = torch.stack([viz_x, viz_rec], 0).view(-1, C, H, W)
			# 	imgs = torchvision.utils.make_grid(imgs, nrow=N).permute(1, 2, 0).unsqueeze(0)
			#
			# 	info = {logger_prefix.format('rec'): imgs.cpu().numpy()}
			#
			# 	try:
			# 		gen = model.generate(N=N).detach()
			# 		imgs = torchvision.utils.make_grid(gen, nrow=N // 2).permute(1, 2, 0).unsqueeze(0)
			#
			# 		info[logger_prefix.format('gen')] = imgs.cpu().numpy()
			#
			# 	except AttributeError:
			# 		pass
			#
			# 	logger.update_images(info, args.total_samples[mode])

			if not silent:
				print('[ {} ] {} Ep={}/{} Itr={}/{} Loss: {:.3f} ({:.3f})'.format(
					time.strftime("%H:%M:%S"), mode,
					epoch + args.start_epoch + 1, args.epochs, i + 1, len(loader),
					stats['loss'].val, stats['loss'].smooth))

		time_stats.update('viz', time.time() - start)
		start = time.time()

	if not silent:
		msg = '[ {} ] {} Ep={}/{} complete Loss: {:.4f} ({:.4f})'.format(
			time.strftime("%H:%M:%S"), mode,
			epoch + args.start_epoch + 1, args.epochs,
			stats['loss'].val, stats['loss'].smooth)
		border = '-' * 50
		print(border)
		print(msg)
		print(border)

	return stats


def run_cls_epoch(model, loader, args, mode='test', optim=None, criterion=None,
                  epoch=None, print_freq=-1, logger=None, unique_tests=False, silent=False,
                  stats_callback=None, stats=None):
	train = mode == 'train'
	if train:
		model.train()
	else:
		model.eval()
	if not hasattr(args, 'total_samples'):
		args.total_samples = {'train': 0, 'test': 0}
	elif mode == 'val' and 'val' not in args.total_samples:
		args.total_samples['val'] = 0

	if not hasattr(args, 'loss_scale'):
		args.loss_scale = 1

	if criterion is None:
		criterion = nn.CrossEntropyLoss()

	print_freq = max(1, len(loader) // 100) if print_freq < 0 else print_freq
	if print_freq == 0:
		print_freq = None

	if stats is None:
		stats = util.StatsMeter(tau=max(min(10 / len(loader), 0.1), 0.01))
	stats.new('loss', 'accuracy', 'confidence')

	time_stats = util.StatsMeter('data', 'fwd', 'bwd', 'viz')
	stats.shallow_join(time_stats, prefix='time-')

	logger_prefix = '{}-{}'.format(mode, '{}') if not unique_tests or train else '{}{}-{}'.format(
		mode, epoch + 1 + args.start_epoch, '{}')

	itr = iter(loader)
	start = time.time()
	for i, sample in enumerate(itr):

		x, y = sample
		if hasattr(args, 'device'):
			x = x.to(args.device)
			y = y.to(args.device)

		args.total_samples[mode] += x.size(0)

		time_stats.update('data', time.time() - start)
		start = time.time()

		pred = model(x)

		loss = criterion(pred, y)

		stats.update('loss', loss)
		loss *= args.loss_scale

		time_stats.update('fwd', time.time() - start)
		start = time.time()

		if train:
			optim.zero_grad()
			loss.backward()
			optim.step()

		time_stats.update('bwd', time.time() - start)
		start = time.time()

		if stats_callback is not None:
			stats_callback(stats, model, sample)

		with torch.no_grad():
			conf, pick = pred.max(-1)

			confidence = conf.detach()
			correct = pick.sub(y).eq(0).float().detach()

			stats.update('confidence', confidence.mean())
			stats.update('accuracy', correct.mean())

		if print_freq is not None and i % print_freq == 0:

			if logger is not None:
				logger.update(stats.smooths(logger_prefix), args.total_samples[mode])

			if not silent:
				print('[ {} ] {} Ep={}/{} Itr={}/{} Loss: {:.3f} ({:.3f})'.format(
					time.strftime("%H:%M:%S"), mode,
					epoch + args.start_epoch + 1, args.epochs, i + 1, len(loader),
					stats['loss'].val, stats['loss'].smooth))

		time_stats.update('viz', time.time() - start)
		start = time.time()

	if not silent:
		msg = '[ {} ] {} Ep={}/{} complete Loss: {:.4f} ({:.4f})'.format(
			time.strftime("%H:%M:%S"), mode,
			epoch + args.start_epoch + 1, args.epochs,
			stats['loss'].val, stats['loss'].smooth)
		border = '-' * 50
		print(border)
		print(msg)
		print(border)

	return stats

