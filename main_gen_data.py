# Example command: python main_pose_rl.py -c config/cartpole/standard.yaml
# nothing
# print('hello')

# Torch imports
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
# torch.multiprocessing.set_sharing_strategy('file_system')

import tensorflow
# print('again')

# Global imports -
import os
import sys
import shutil
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import h5py as hf

# Local imports
import foundation as fd
import foundation.util as util

# import resource
#
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2 ** 16, rlimit[1]))
# print('3')

def init_worker(seed, render_target=False, moving_target=False, img_shape=(128,128), **other_args):
	np.random.seed(seed)
	torch.manual_seed(seed)
	output = {'env': fd.envs.Arm(render_target=render_target, moving_target=moving_target)}
	output['env']._render_size = img_shape
	return output

def create_sample(env, policy, x_0, dx_0, num_idx, set_idx, seq_len, step_len=1, lengths=None, masses=None, **other_args):

	sample = {'controls', 'states', 'rgbs', 'labels'}
	sample = {k:[] for k in sample}

	if lengths is not None:
		env.set_lengths(lengths[set_idx])
	if masses is not None:
		env.set_masses(masses[set_idx])

	env.reset(x_0=x_0[set_idx, num_idx], dx_0=dx_0[set_idx, num_idx])
	state = torch.cat([env.sim.u, env.sim.du])
	sample['states'].append(state)

	rgb0, label0 = env.render(label=True)

	sample['rgbs'].append(util.rgb_to_str(rgb0))
	sample['labels'].append(util.byte_img_to_str(label0.astype(np.uint8)))

	for i in range(seq_len):
		action = policy(i, state)
		for _ in range(step_len):
			env.step(action)
		state = torch.cat([env.sim.u, env.sim.du])
		sample['controls'].append(action)
		sample['states'].append(state)

		rgb, label = env.render(label=True)

		sample['rgbs'].append(util.rgb_to_str(rgb))
		sample['labels'].append(util.byte_img_to_str(label.astype(np.uint8)))

	sample['states'] = torch.stack(sample['states']).numpy()
	sample['controls'] = torch.stack(sample['controls']).numpy()

	sample['lengths'] = env.lengths.numpy()
	sample['masses'] = env.masses.numpy()

	return set_idx, num_idx, sample

################ MAIN
def main():
	global args
	parser = util.setup_gen_data_options()
	args = parser.parse_args()

	args.move_target = False
	args.render_target = False
	args.vary_masses = True
	args.vary_lengths = False

	print('Generating arm episodes (change env settings manually)')

	env = fd.envs.Arm()
	size = env.spec.action_space.size

	assert args.policy == 'random', 'only uniform random supported currently'
	policy = lambda t, x: torch.rand(size)*2 - 1
	print('Policy: uniform random')

	# initial conditions
	if args.move_target:
		assert False, 'moving target not curerntly supported'
		x_0 = torch.rand(args.sets, args.num, env.N + 2) * 2 * np.pi
		x_0[:, :, env.N] = torch.sqrt(torch.rand(args.sets, args.num))

		dx_0 = torch.randn(env.N + 2) * 6 + 6
		dx_0[env.N:env.N + 2] = torch.randn(2) + 0.5
		dx_0[torch.rand(env.N + 2) < 0.5] *= -1
		dx_0[env.N + 1] *= 2

	else:
		x_0 = torch.rand(args.sets, args.num, env.N) * 2 * np.pi

		dx_0 = torch.randn(args.sets, args.num, env.N) * 6 + 6
		dx_0[torch.rand(args.sets, args.num, env.N) < 0.5] *= -1


	# vary params
	lengths = None
	if args.vary_lengths:
		lims = torch.Tensor([0.4, 0.6])
		r = lims[1] - lims[0]
		lengths = torch.rand(args.sets, 1) * r + lims[0] # uniform distribution
		lengths = torch.cat([lengths, 1 - lengths], -1)

	masses = None
	if args.vary_masses:
		lims = torch.Tensor([0.5, 2]).log()
		r = lims[1] - lims[0]
		masses = torch.rand(args.sets, 1) * r + lims[0]
		masses = masses.exp() # log uniform distribution
		masses = torch.cat([torch.ones(args.sets, 1), masses], -1)

	if args.view_init_cond:
		pos = x_0[:, :, :2].view(-1, 2).numpy()
		vel = dx_0[:, :, :2].view(-1, 2).numpy()

		plt.figure()
		plt.title('init pos distribution')
		plt.xlabel('angle (rad)')
		plt.ylabel('counts')
		plt.hist(pos)

		plt.figure()
		plt.title('init vel distribution')
		plt.xlabel('angular vel (rad/sec)')
		plt.ylabel('counts')
		plt.hist(vel)

		print('Pos: {} +/- {} (min={},max={})'.format(str(pos.mean(0).round(3)), str(pos.std(0).round(3)), str(pos.min(0).round(3)), str(pos.max(0).round(3))))
		print('Vel: {} +/- {} (min={},max={})'.format(str(vel.mean(0).round(3)), str(vel.std(0).round(3)), str(vel.min(0).round(3)), str(vel.max(0).round(3))))

		if args.vary_lengths:
			ls = lengths.numpy()
			plt.figure()
			plt.title('lengths distribution')
			plt.xlabel('length')
			plt.ylabel('counts')
			plt.hist(ls)

			print('Lengths: {} +/- {} (min={},max={})'.format(str(ls.mean(0).round(3)), str(ls.std(0).round(3)),
														  str(ls.min(0).round(3)), str(ls.max(0).round(3))))

		if args.vary_masses:
			ms = masses.numpy()
			plt.figure()
			plt.title('masses distribution')
			plt.xlabel('mass')
			plt.ylabel('counts')
			plt.hist(ms)

			print('Masses: {} +/- {} (min={},max={})'.format(str(ms.mean(0).round(3)), str(ms.std(0).round(3)),
														  str(ms.min(0).round(3)), str(ms.max(0).round(3))))


		plt.show()

	def gen_idx(total_sets, total_num):
		for si, ni in np.ndindex(total_sets, total_num):
			yield {'set_idx':si, 'num_idx':ni}

	args.img_wd, args.img_ht = 128, 128

	creator = util.Farmer(create_sample,
		shared_args={
			'x_0': x_0,
			'dx_0':dx_0,
			'lengths':lengths,
			'masses':masses
		}, private_args={
			'moving_target': args.move_target,
			'render_target': args.render_target,
			'policy': policy,
			'seq_len': args.seq_len,
			'step_len': args.step_len,
			'img_shape': (args.img_wd, args.img_ht),
		},
		unique_worker_args=[{'seed':s} for s in range(args.num_workers)],
		volatile_gen=gen_idx(args.sets, args.num),
		init_fn=init_worker, num_workers=args.num_workers)

	print_step = max(args.num // 100, 1)

	dt = util.AverageMeter()

	args.save += '_{}.h5'

	samples = [deque() for _ in range(args.sets)]
	completed = 0

	img_set = {'rgb', 'labels'}

	start = time.time()

	for itr, (set_idx, num_idx, sample) in enumerate(creator):

		dt.update(time.time() - start)
		#if itr % print_step == 0:
		#	print('{}/{} complete, progress={}/{}, Timing: {:.5f} ({:.5f}) s/ep'.format(completed, args.sets, [len(x) if x is not None else args.num for x in samples], args.num, dt.val, dt.avg))

		if itr % print_step == 0:
			print('{}/{} complete, Timing: {:.5f} ({:.5f}) s/ep'.format(completed, args.sets, dt.val, dt.avg))


		samples[set_idx].append(sample)

		if len(samples[set_idx]) == args.num:

			complete_set = samples[set_idx]

			save_dir = args.save.format(set_idx)

			with hf.File(save_dir, 'w') as f:
				for key in complete_set[0].keys():
					if key in img_set:
						f.create_dataset(key, data=[sample[key] for sample in complete_set])
					elif key == 'lengths':
						assert np.abs((sum([sample[key] for sample in complete_set])/args.num - complete_set[0][key]).sum()) < 1e-5
						f.create_dataset(key, data=complete_set[0][key])
					elif key == 'masses':
						if not np.abs((sum([sample[key] for sample in complete_set])/args.num - complete_set[0][key]).sum()) < 1e-5:
							print([sample[key] for sample in complete_set])
							quit()
						f.create_dataset(key, data=complete_set[0][key])
					else:
						f.create_dataset(key, data=np.stack([sample[key] for sample in complete_set]))

			print('-- Set {}/{} complete, saved to {} --'.format(set_idx+1, args.sets, save_dir))
			completed += 1
			del complete_set
			samples[set_idx] = None

		start = time.time()

if __name__ == '__main__':
	main()

