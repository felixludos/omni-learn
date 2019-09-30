
import sys, os
import time
import numpy as np
import gym
import torch
import torch.nn as nn
from torch import Tensor
#import matplotlib.pyplot as plt
import configargparse
from tabulate import tabulate
import torch.multiprocessing as mp

import foundation as fd
import foundation.util as util
from foundation.data.samplers import Trajectory_Generator
from foundation.envs import Policy_Evaluator

args = None # args

def main(argv=None):
	
	global args
	if argv is not None:
		args = argv
	else:
		parser = util.options.setup_multi_options()
		args = parser.parse_args()

	assert args.name is not None or args.resume is not None, 'Must specify a name when not resuming'

	if args.config is None and args.resume is not None: # replace args with
		resume, ckpt = util.load_checkpoint(args.resume)
		args = ckpt['args']
		args.resume = resume
		print('Loaded args from {}'.format(args.resume))

	args.cuda = args.cuda and torch.cuda.is_available()
	args.def_type = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'
	if args.cuda:
		print('Using CUDA')

	manager, env = util.setup_multi_system(args)

	total_episodes = 0
	train_stats = util.StatsMeter() # TODO: use global stats, and append new ones to global
	eval_stats = util.StatsMeter()
	time_stats = util.StatsMeter('learning', 'eval', 'meta')
	best_perf = None
	
	# TODO: setup logging before init model/env
	
	assert not (args.load is not None and args.resume is not None), 'Cant both load and resume'
	
	load_path = args.load if args.load is not None else args.resume
	if load_path is not None:
		load_path, ckpt = util.load_checkpoint(load_path)

		manager.load_state_dict(ckpt['agent'])
		train_stats.load_dict(ckpt['train-stats'])
		eval_stats.load_dict(ckpt['eval-stats'])
		load_args = ckpt['args']
		args = util.check_load_args(args, load_args)
		try:
			best_perf = ckpt['best']
		except:
			print('No saved best perf')
		total_episodes = ckpt['total-episodes']
		if total_episodes is None:
			total_episodes = 0
		
		print('Loaded checkpoint: {}'.format(load_path))
	
	now = time.strftime("%b-%d-%Y-%H%M%S")
	if args.resume is not None:
		args.save_dir = os.path.dirname(load_path)
	else:
		args.save_dir = os.path.join(args.save_root, args.name, now)
		util.create_dir(args.save_dir)
	print('Saving checkpoints to: {}'.format(args.save_dir))
	
	# setup saving stuff
	statlogger = None
	tblogger = None
	if args.log:
		tblogger = util.TBLogger(args.save_dir)  # Start tensorboard logger
		tblogger.scalar_summary('zzz-ignore', 0, 0)
		# Create logfile to save prints
		logfile = open(os.path.join(args.save_dir, 'logfile.txt'), 'a+')
		backup = sys.stdout
		sys.stdout = util.Tee(sys.stdout, logfile)

		statlogger = util.StatsCollector() if args.save_stats else None
		if statlogger is not None:
			print('Logging stats')

	if args.resume is not None:
		print('--- Resuming on {} ---'.format(now))
	
	print('Env: {} ({} agents):'.format(args.env, len(env.spec)))
	for spec in env.spec:
		print('\t(obs={}, act={})'.format(spec.obs_space.size, spec.action_space.size))
	
	# if args.model == 'mlp':
	# 	print('Cmd Model: {}'.format(manager.cmd_agent.policy.model))
	# 	print('Sub Model: {}'.format(manager.sub_agents[0].policy.model))
	# if args.baseline == 'mlp':
	# 	print('Cmd Baseline: {}'.format(manager.cmd_agent.baseline.model))
	# 	print('Sub Baseline: {}'.format(manager.sub_agents[0].baseline.model))
	
	if args.seed is None:
		args.seed = util.get_random_seed()
	print('Seed: {}'.format(args.seed))
	
	print('Training: num-iter={}, num-traj={}'.format(args.num_iter, args.num_traj))
	
	path_gen = Trajectory_Generator(N=args.num_iter * args.num_traj, num_workers=args.num_gen_workers,
	                                batch_size=args.num_traj, multi_agent=True,
	                                policy=manager.policy, args=args, seed=args.seed, def_type=args.def_type,
	                                cache=args.cache_iter)
	
	evaluator = Policy_Evaluator(num_workers=args.num_eval_workers, N=args.num_eval, policy=manager.policy, args=args,
	                             seed=args.seed + args.num_gen_workers)
	
	for itr, paths in enumerate(path_gen):

		new_best = False
		if args.verbose:
			print('.' * 50)
			print('ITERATION {}/{} : episode={}'.format(itr + 1, args.num_iter, total_episodes))
		
		start = time.time()

		# learn from paths
		train_stats = manager.train_step(paths)
		for infos in paths[0].env_infos:
			for k, v in infos.items():
				if k not in train_stats:
					train_stats.new(k)
				train_stats.update(k, v.mean().item())

		time_stats.update('learning', time.time() - start)
		start = time.time()
		# evaluate policy producing eval_stats
		if args.num_eval > 0:
			# set best_perf
			manager.eval()
			estats = next(evaluator)
			eval_stats = estats[0]
			for estat in estats[1:]:
				eval_stats.join(estat)
			manager.train()
			new_best = best_perf is None or eval_stats['perf'].avg >= best_perf
			if new_best: best_perf = eval_stats['perf'].avg

		time_stats.update('eval', time.time() - start)
		start = time.time()

		# print results
		if args.verbose:
			# print stats TODO: using tabulate
			print('Training:')
			print(tabulate(sorted([(name, stat.val, stat.avg) for name, stat in train_stats.items()])))
			print('Eval:')
			print(tabulate(sorted([(name, stat.val, stat.avg) for name, stat in eval_stats.items()])))
			print('Timing:')
			print(tabulate(sorted([(name, stat.val, stat.avg) for name, stat in time_stats.items()])))
		else: # condensed print
			print("[ {} ] {:}/{:} (ep={}) : {:5.3f} {:5.3f} {:5.3} ".format((time.strftime("%m-%d-%y %H:%M:%S")),itr+1, args.num_iter, total_episodes,
																   train_stats['returns'].avg, eval_stats['perf'].avg, best_perf if best_perf is not None else ''))
		# save checkpoint
		#total_episodes += len(paths)
		total_episodes += args.num_traj
		if new_best:
			saved_path = util.save_checkpoint(args, manager, logger=None, episode=total_episodes, best=best_perf, is_best=True)
			print('--- New Best Model Saved to {} ---'.format(saved_path))

		if itr % args.save_freq == 0:
			saved_path = util.save_checkpoint(args, manager, logger=statlogger, episode=total_episodes, best=best_perf, is_best=False)
			print('--- Checkpoint Saved to {} ---'.format(saved_path))

		time_stats.update('meta', time.time() - start)

		if tblogger is not None: # update tensorboard
			info = {'atrain-'+name : train_stats[name].avg for name in train_stats.keys()}
			info.update({'eval-'+name : eval_stats[name].avg for name in eval_stats.keys()})
			info.update({'timing-' + name: time_stats[name].val for name in time_stats.keys()})
			for k, v in info.items():
				tblogger.scalar_summary(k, v, total_episodes)


			if statlogger is not None:
				statlogger.update(total_episodes, info)


	# Final Save
	saved_path = util.save_checkpoint(args, manager, logger=statlogger, episode=total_episodes, best=best_perf, is_best=False)
	print('--- Final Checkpoint Saved to {} ---'.format(saved_path))
	
	# visualize policy
	if args.viz > 0:
		print('Rendering {} episodes'.format(args.viz))
		env.visualize_policy(N=args.viz, policy=manager.policy)

	print('Job Complete.')

if __name__ == '__main__':
	mp.freeze_support()
	mp.set_start_method('spawn')
	main()


