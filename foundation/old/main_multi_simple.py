
import sys, os
import time
import numpy as np
import gym
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import configargparse

import envs
import agents
import util
import models
import options

args = None

def check_load_args(load_args):
	assert load_args.alg == args.alg
	assert load_args.env == args.env
	assert load_args.policy == args.policy
	args = new_args
	return new_args

def log_stats(name, stats, n=None):
	for stat in stats.keys():
		args.tblogger.scalar_summary(name + '-' + stat, stats[stat].val, n)

def save_checkpoint(agent, train_stats, eval_stats, episode=None, is_best=False):
	assert episode or is_best
	name = 'model_best.pth.tar' if is_best else 'checkpoint_{}.pth.tar'.format(episode)
	save_path = os.path.join(args.save_dir, name)
	ckpt = {
		'agent': agent.save(),
		'train-stats': train_stats.export(),
		'eval-stats': eval_stats.export(),
		'args': args,
		'total-episodes': episode,
	}
	torch.save(ckpt, save_path)
	if is_best:
		print('--- New Best Model Saved to {} ---'.format(name))
	else:
		print('--- Checkpoint Saved to {} ---'.format(name))

def print_stats(episode, **stats):
	print('Episode {episode}: \n\tloss = {loss.val:.5f} ({loss.avg:.5f}) \n\treward = {reward.val:.5f} ({reward.avg:.5f})'.format(episode=episode, **stats))

def evaluate(env, agent, episodes): # returns rewards and episode lengths
	stats = util.StatsMeter('reward', 'steps')
	agent.eval()

	for i in range(episodes):
		state = env.reset()
		test_reward = 0
		test_steps = 0

		for _ in range(env.spec.timestep_limit):
			action = agent.get_action(state)
			state, reward, done, _ = env.step(action) # WARNING: scaling actions
			test_reward += reward
			test_steps += 1
			if done: break
		stats.update('steps', test_steps)
		stats.update('reward', test_reward / test_steps)
		log_stats('eval', stats)
	return stats

def main(argv=None):
	
	global args
	if argv is not None:
		args = argv
	else:
		parser = options.setup_discrete_options()
		args = parser.parse_args()

	# init env
	if args.env == 'discrete':
		env = envs.Discrete_Pos_Trap(n_agents=args.num_agents, n_particles=args.num_particles,
									 grid_side=args.grid_side, obs_grid=args.obs_grid)
		args.action_dim = env.action_options
		args.obs_shape = env.observation_space.shape

	elif args.env == 'pendulum':
		#assert args.policy == 'small'
		env = gym.make('Pendulum-v0')
		env = env.env
		args.action_dim = env.action_space.shape[0]
		args.obs_shape = env.observation_space.shape
	
	elif args.env == 'number-game':
		env = envs.Coop_Guess_Number(two_way=True, separate_rewards=args.separate_rewards)
		
		args.n_agents = env.obs_space.shape[0]
		args.obs_dim = env.obs_space.shape[1]
		args.action_dim = env.action_space[1]
	else:
		raise Exception('Unknown env name: {}'.format(args.env))

	print('Using {} env (obs={},action={})'.format(args.env, args.obs_shape, args.action_dim))

	args.cuda = args.cuda and torch.cuda.is_available()
	args.def_type = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'

	# init model
	if args.policy == 'branched':
		model = models.Branched_Net(args.obs_shape, args.action_dim, hidden_dims=[240, 200, 160,], value_dims=[40, 20], action_dims=[90, 60], 
					nonlinearity=args.nonlinearity, use_bn=args.use_bn)
	elif args.policy == 'branched-small':
		model = models.Branched_Net(args.obs_shape, args.action_dim, hidden_dims=[20,30], value_dims=[20, 10], action_dims=[20, 10], 
					nonlinearity=args.nonlinearity, use_bn=args.use_bn)
	elif args.policy == 'separate': 
		model = models.Separate_Net(obs_dim=int(np.product(args.obs_shape)), action_dim=args.action_dim, 
									critic_hidden_dims=[30, 30], actor_hidden_dims=[30, 30], nonlinearity='elu', use_bn=args.use_bn)
	elif args.policy == 'tiny':
		model = models.Policy_Net(obs_dim=args.obs_dim, action_dim=args.action_dim,
		                         hidden_dims=[5,5], nonlinearity='elu', output_nonlin='sigmoid')
	else:
		raise Exception('Unknown policy name: {}'.format(args.policy))

	# init alg
	if args.alg == 'npg':
		raise Exception('not implemented')
	elif args.alg == 'ddpg':
		agent_fn = agents.DDPG
	else:
		raise Exception('Unknown alg name: {}'.format(args.alg))

	agent = agent_fn(model=model, args=args)

	total_episodes = 0

	# train and eval
	train_stats = util.StatsMeter('loss', 'steps')
	eval_stats = util.StatsMeter('reward', 'steps')
	best = None

	# load previous
	now = time.strftime("%b-%d-%Y-%H%M%S")
	if args.resume is not None:
		if os.path.isdir(args.resume):
			args.resume = os.path.join(args.resume, 'model_best.pth.tar')
		
		ckpt = torch.load(args.resume)
		agent.load(checkpoint=ckpt['agent'])
		train_stats.join(ckpt['train-stats'])
		eval_stats.join(ckpt['eval-stats'])
		load_args = ckpt['args']
		args = check_load_args(load_args)
		total_episodes = ckpt['total-episodes']

		args.save_dir = os.path.dirname(args.resume)
		print('Loaded checkpoint: {}'.format(args.resume))
	else:
		args.save_dir = os.path.join(args.save_root, args.name + '_' + now)
		util.create_dir(args.save_dir)
		print('Saving checkpoints to: {}'.format(args.save_dir))
	
	args.tblogger = util.TBLogger(args.save_dir)  # Start tensorboard logger
	args.tblogger.scalar_summary('zzz-ignore', 0, 0)
	# Create logfile to save prints
	logfile = open(os.path.join(args.save_dir, 'logfile.txt'), 'a+')
	backup = sys.stdout
	sys.stdout = util.Tee(sys.stdout, logfile)

	if args.resume is not None:
		print('Resuming on {}'.format(now))

	for ep in range(args.episodes):
		
		# train
		total_episodes += 1

		train_steps = 0
		train_loss = 0

		state = env.reset()
		
		agent.train()
		for step in range(env.spec.timestep_limit):

			#env.render()

			action = agent.get_action(state)

			next_state, reward, done, _ = env.step(action)

			loss = agent.learn(state, action, reward, next_state, done)

			if loss is not None:
				train_steps += 1
				train_loss += loss

			state = next_state
			if done: break

		if train_steps:
			train_stats.update('loss', train_loss / train_steps)
			train_stats.update('steps', train_steps)
			log_stats('train', train_stats)

		# Test
		if total_episodes and args.test_freq and total_episodes % args.test_freq == 0:
			new_eval_stats = evaluate(env, agent, args.test_len)
			eval_stats.join(new_eval_stats)
			if best is None or new_eval_stats['reward'].avg > best:
				best = new_eval_stats['reward'].avg
				save_checkpoint(agent, train_stats, eval_stats, episode=total_episodes, is_best=True)

		if total_episodes and args.print_freq and total_episodes % args.print_freq == 0:
			print_stats(total_episodes, loss=train_stats['loss'], reward=eval_stats['reward'])

		# Save
		if total_episodes and total_episodes % args.save_freq == 0:
			save_checkpoint(agent, train_stats, eval_stats, episode=total_episodes)

	# Final Save
	save_checkpoint(agent, train_stats, eval_stats, episode=total_episodes)


if __name__ == '__main__':
	main()


