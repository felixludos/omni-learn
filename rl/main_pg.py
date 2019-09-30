
import sys, os
import time
import numpy as np
import gym
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import configargparse
from tabulate import tabulate
import torch.multiprocessing as mp

import foundation as fd
from foundation import util
from foundation import models
from foundation import rl
from foundation import envs
from foundation import train

class NormalizedMLP(fd.Supervised_Model):
	def __init__(self, input_dim, output_dim, norm=True, **args):
		super().__init__(in_dim=input_dim, out_dim=output_dim, criterion=util.get_loss_type('mse'))
		self.norm = models.RunningNormalization(input_dim) if norm else None
		self.net = models.make_MLP(input_dim, output_dim, **args)

	def forward(self, x):
		if self.norm is not None:
			x = self.norm(x)
		return self.net(x)




def main(args=None):
	if args is None:
		parser = train.setup_rl_options()
		args = parser.parse_args()

	now = time.strftime("%y-%m-%d-%H%M%S")
	if args.log_date:
		args.name = os.path.join(args.name, now)
	args.save_dir = os.path.join(args.save_root, args.name)
	print('Save dir: {}'.format(args.save_dir))
	if args.log_tb or args.log_txt or args.save_freq is not None:
		util.create_dir(args.save_dir)
		print('Logging/Saving in {}'.format(args.save_dir))
	logger = util.Logger(args.save_dir, tensorboard=args.log_tb, txt=args.log_txt)

	if args.seed is None:
		args.seed = util.get_random_seed()
		print('Generating random seed: {}'.format(args.seed))

	torch.manual_seed(args.seed)
	print('Using {}'.format(args.device))

	env = envs.Pytorch_Gym_Env(args.env, device=args.device)
	env.seed(args.seed)

	args.state_dim, args.action_dim = len(env.observation_space.low), len(env.action_space.low)

	n_batch = args.budget_steps / args.steps_per_itr

	if 'mlp' in args.baseline:

		baseline_model = NormalizedMLP(args.state_dim, 1, norm='norm' in args.baseline,
		                               hidden_dims=args.b_hidden, nonlin=args.b_nonlin)

		baseline_model.optim = util.get_optimizer(args.b_optim_type, baseline_model.parameters(), lr=args.b_lr, weight_decay=args.b_weight_decay)
		baseline_model.scheduler = torch.optim.lr_scheduler.LambdaLR(
					baseline_model.optim, lambda x: (n_batch - x) / n_batch, -1)

		# print(baseline_model.optim)
		# quit()

		#assert args.baseline == 'norm-mlp'
		baseline = rl.Deep_Baseline(baseline_model, scale_max=args.b_scale_max,
		                    batch_size=args.b_batch_size, epochs_per_step=args.b_epochs, )

	elif args.baseline == 'lin':
		baseline = rl.Linear_Baseline(state_dim=args.state_dim, value_dim=1)
	else:
		raise Exception('unknown baseline: {}'.format(args.baseline))

	assert args.policy == 'normal'
	assert args.model == 'norm-mlp'
	policy_model = NormalizedMLP(args.state_dim, 2 * args.action_dim, hidden_dims=args.hidden, nonlin=args.nonlin)
	policy = rl.NormalPolicy(policy_model, )

	assert args.agent == 'ppoclip'

	agent = rl.PPOClip(policy=policy, baseline=baseline, clip=args.clip, normalize_adv=args.norm_adv,
	            optim_type=args.optim_type, lr=args.lr, scheduler_lin=n_batch, weight_decay=args.b_weight_decay,
	            batch_size=args.batch_size, epochs_per_step=args.epochs,
	            ).to(args.device)

	print(agent)
	print('Agent has {} parameters'.format(util.count_parameters(agent)))

	gen = fd.data.Generator(env, agent, step_limit=args.budget_steps,
	                step_threshold=args.steps_per_itr, drop_last_state=True)

	train.run_rl_training(gen, agent, args=args, logger=logger, save_freq=args.save_freq)

if __name__ == '__main__':
	#mp.freeze_support()
	#mp.set_start_method('spawn')
	main()


