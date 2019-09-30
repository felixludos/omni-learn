
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import gym
import h5py as hf
import configargparse
from collections import deque

import foundation as fd
from foundation import util
from foundation import nets
from foundation import train


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2 ** 16, rlimit[1]))

def save_complete_set(complete_set, save_dir):
	with hf.File(save_dir, 'w') as f:
		for key in complete_set[0].keys():
			f.create_dataset(key, data=np.stack([sample[key].numpy() for sample in complete_set]))

def init_gym_worker(seed, env_id, **other_args):
	np.random.seed(seed)
	torch.manual_seed(seed)
	output = {'env': util.Numpy_Env_Wrapper(gym.make(env_id))}
	return output

# def init_worker(seed, env_type, env_args, **other_args):
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	output = {'env': env_type(**env_args)}
# 	return output

def create_gym_sample(env, policy, num_idx, set_idx, seq_len, step_len=1,
                  **other_args):
	
	sample = {'controls', 'states', 'rewards'}
	sample = {k :[] for k in sample}
	
	state = env.reset()
	sample['states'].append(state)
	
	for i in range(seq_len):
		action = policy(i, state)
		for _ in range(step_len):
			state, reward, _, _ = env.step(action)
		sample['controls'].append(action)
		sample['states'].append(state)
		sample['rewards'].append(reward)
	
	sample['states'] = torch.stack(sample['states'])
	sample['controls'] = torch.stack(sample['controls'])
	sample['rewards'] = torch.tensor(sample['rewards']).float()
	
	return set_idx, num_idx, sample

def main():
	global args
	
	parser = configargparse.ArgumentParser(description='Generate data of')
	
	parser.add_argument('--num', type=int, default=10, help='Number of sequences to generate for each set')
	parser.add_argument('--sets', type=int, default=1, help='Number of sets to generate')
	parser.add_argument('--set-offset', type=int, default=0, help='Offset for dataset naming')
	parser.add_argument('-s', '--save', default='results', type=str, metavar='PATH',
	                    help='path to save results in (w/out suffix)')
	
	parser.add_argument('--seq-len', default=None, type=int,
	                    metavar='N', help='length of the training sequence (default: 1)')
	parser.add_argument('--step-len', default=1, type=int, metavar='N',
	                    help='number of frames separating each example in the training sequence (default: 1)')
	
	parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
	                    help='number of data loading workers (default: 4)')
	parser.add_argument('--img-ht', default=10, type=int,
	                    metavar='N', help='img height')
	parser.add_argument('--img-wd', default=10, type=int, metavar='N',
	                    help='img width')
	
	parser.add_argument('--policy', default='random', type=str, help='pre-defined policy to be used to choose control')
	parser.add_argument('--env', default='arm', type=str, help='env for which to collect data')
	
	args = parser.parse_args()
	
	
	
	if args.env in {'Pendulum-v0'}:
		pass
	else:
		raise Exception('Unknown env: {}'.format(args.env))
	
	
	def gen_idx(total_sets, total_num):
		for si, ni in np.ndindex(total_sets, total_num):
			yield {'set_idx' :si, 'num_idx' :ni}
	
	
	env = gym.make(args.env)
	
	if args.seq_len is None:
		args.seq_len = env.spec.timestep_limit
	
	size = env.observation_space.shape[0]
	assert args.policy == 'random', 'only uniform random supported currently'
	policy = lambda t, x: torch.rand(size) * 2 - 1
	print('Policy: uniform random')
	
	
	shared_args = {
	
	}
	private_args = {
		'policy': policy,
		'seq_len': args.seq_len,
		'step_len': args.step_len,
		'env_id': args.env,
	}
	
	
	creator = util.Farmer(create_gym_sample,
	                 shared_args=shared_args, private_args=private_args,
	                 unique_worker_args=[{'seed' :s} for s in range(args.num_workers)],
	                 volatile_gen=gen_idx(args.sets, args.num),
	                 init_fn=init_gym_worker, num_workers=args.num_workers)
	
	print_step = max(args.num * args.sets // 1000, 1)
	
	dt = util.AverageMeter()
	
	args.save += '_{}.h5'
	
	samples = [deque() for _ in range(args.sets)]
	completed = 0
	
	start = time.time()
	
	for itr, (set_idx, num_idx, sample) in enumerate(creator):
		
		dt.update(time.time() - start)
		
		if itr % print_step == 0:
			print('{}/{} complete, {}/{} num, Timing: {:.5f} ({:.5f}) s/ep'.format(completed, args.sets ,
			        max([len(x) if x is not None else 0 for x in samples]), args.num,  dt.val.item(), dt.smooth.item()))
		
		samples[set_idx].append(sample)
		
		if len(samples[set_idx]) == args.num:
			save_dir = args.save.format(set_idx)
			save_complete_set(samples[set_idx], save_dir)
			
			print('-- Set {}/{} complete, saved to {} --'.format(set_idx+1, args.sets, save_dir))
			completed += 1
			samples[set_idx] = None
		
		start = time.time()



if __name__ == '__main__':
	main()





