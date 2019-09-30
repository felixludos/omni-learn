import sys, os
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.distributions as distrib
import torch.nn as nn
import torch.utils.data as td
from torch.utils.data import Dataset, DataLoader
import gym
import numpy as np
#%matplotlib notebook
import matplotlib.pyplot as plt
plt.switch_backend('TKAgg') #('Qt5Agg')
from scipy.sparse import coo_matrix
#from envs import Discrete_Pos_Trap, Gym_Env, Print_Env
from torch.distributions.utils import lazy_property
import foundation as fd
from foundation.util import NS
import ray
#import sequential_ray as ray
from ray_rl_backend import *


args = None # args

def main(argv=None):
	
	global args
	
	if argv is not None:
		args = argv
	else:
		parser = util.options.setup_ray_options()
		args = parser.parse_args()
		
	args.def_type = 'torch.FloatTensor'
	assert args.name is not None or args.resume is not None, 'Must specify a name when not resuming'
	ray.init()

	print('Name: {}'.format(args.name))
	
	
	env = util.get_env(args.env)
	obs_dim = env.spec.obs_space.size
	act_dim = env.spec.action_space.choices if args.env == 'cartpole' else env.spec.action_space.size
	print('Env: {} (obs={},act={})'.format(args.env, obs_dim, act_dim))
	
	model = fd.nets.MLP(input_dim=obs_dim, output_dim=act_dim, hidden_dims=args.hidden, nonlinearity=args.nonlinearity)
	print(model)
	if args.policy == 'gaussian':
		policy = fd.policies.Gaussian_Policy(model=model, def_type=args.def_type, init_log_std=0, min_log_std=-3)
	elif args.policy == 'cat':
		policy = fd.policies.Categorical_Policy(model=model, def_type=args.def_type)
	else:
		raise Exception('unknown policy: {}'.format(args.policy))
	baseline = None
	
	score = None
	stats = util.StatsMeter('score')
	time_stats = util.StatsMeter('traj', 'train', 'eval')
	best_perf = None
	new_best = False
	
	load_path = args.load if args.load is not None else args.resume
	if load_path is not None:
		load_path, ckpt = load_checkpoint(load_path)
		
		policy.load_state_dict(ckpt['policy'])
		baseline = ckpt['baseline']
		stats.load_dict(ckpt['stats'])
		load_args = ckpt['args']
		args = util.check_load_args(args, load_args)
		best_perf = ckpt['best']
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
	tblogger = None
	if args.log:
		tblogger = util.TBLogger(args.save_dir)  # Start tensorboard logger
		tblogger.scalar_summary('zzz-ignore', 0, 0)
		# Create logfile to save prints
		logfile = open(os.path.join(args.save_dir, 'logfile.txt'), 'a+')
		backup = sys.stdout
		sys.stdout = util.Tee(sys.stdout, logfile)
	
	if args.resume is not None:
		print('--- Resuming on {} ---'.format(now))
	
	total_episodes = 0
	
	seed = args.seed
	util.set_seed(seed)
	
	for itr in range(args.num_iter):
		start = time.time()
		
		# generate paths
		paths = [generate_path.remote(policy, env, baseline=baseline, discount=args.gamma, seed=i + seed, obs_order=args.obs_order,
							   time_order=args.time_order)
				 for i in range(args.num_traj)]
		seed += args.num_traj
		paths = collate(ray.get(paths))
		time_stats.update('traj', time.time() - start)
		start = time.time()
		
		# train baseline
		baseline_id, bsln_stats_id = train_baseline.remote(baseline, paths, reg_coeff=args.reg_coeff, obs_order=args.obs_order, time_order=args.time_order)
		
		# train policy
		policy_id, optim_stats_id = npg_update.remote(policy, paths, def_type=args.def_type, reg_coeff=args.reg_coeff,
										 step_size=args.delta, max_cg_iter=args.max_steps)
		
		policy = ray.get(policy_id)
		stats.join(ray.get(optim_stats_id), prefix='optim-')
		score = stats['optim-returns'].avg if score is None else score * 0.9 + stats['optim-returns'].avg * 0.1
		stats.update('score', score)
		total_episodes += args.num_traj
		
		time_stats.update('train', time.time() - start)
		start = time.time()
		
		# eval
		eval_stats_ids = [evaluate_policy.remote(policy, env, seed=i + seed) for i in range(args.num_eval)]
		seed += args.num_eval
		for estat_id in eval_stats_ids:
			stats.join(ray.get(estat_id), prefix='eval-')
		time_stats.update('eval', time.time() - start)
		
		if best_perf is None or stats['eval-reward'].avg >= best_perf:
			new_best = True
			best_perf = stats['eval-reward'].avg
		
		stats.join(ray.get(bsln_stats_id), prefix='bsln-')
		baseline = ray.get(baseline_id)
		
		print_stats(stats, time_stats, best_perf=best_perf, itr=itr, num_iter=args.num_iter, logger=tblogger,
					total_episodes=total_episodes, verbose=args.verbose)
		
		if new_best:
			save_path = os.path.join(args.save_dir, 'model_best.pth.tar')
			save_checkpoint.remote(save_path, args, policy, baseline, stats, episode=total_episodes, best=best_perf)
			print('--- New Best Model Saved to {} ---'.format(save_path))

		if itr % args.save_freq == 0:
			save_path = os.path.join(args.save_dir, 'checkpoint_{}.pth.tar'.format(total_episodes))
			save_checkpoint.remote(save_path, args, policy, baseline, stats, episode=total_episodes, best=best_perf)
			print('--- Checkpoint Saved to {} ---'.format(save_path))
			
		new_best = False

if __name__ == '__main__':
	main()


