
import sys, os
import time
import numpy as np
import gym
import torch
#import matplotlib.pyplot as plt
import configargparse
import foundation.util as util

args = None # args

def main(argv=None):
	
	global args
	if argv is not None:
		args = argv
	else:
		parser = util.options.setup_viz_options()
		args = parser.parse_args()

	args.path, ckpt = util.load_checkpoint(args.path)
	largs = ckpt['args']

	print('Loaded {}'.format(args.path))
	
	if 'multi_agent' in largs and largs.multi_agent:
		agent, env = util.setup_multi_system(largs)
	else:
		agent, env = util.setup_system(largs)

	agent.load_state_dict(ckpt['agent'])
	#print('Env: {} (obs={}, act={})'.format(largs.env, env.spec.obs_space.size, env.spec.action_space.size))
	#print('Agent: {} (policy={}, model={}, baseline={})'.format(largs.agent, largs.policy, largs.model, largs.baseline))

	agent.eval()

	print('Rendering {} episodes'.format(args.n))
	#print(type(env))
	env.visualize_policy(policy=agent.policy, N=args.n)

	print('Job Complete.')

if __name__ == '__main__':
	main()


