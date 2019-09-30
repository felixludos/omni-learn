import sys, os
import numpy as np
import time as timer
import pickle
import copy
import time
from tabulate import tabulate
import torch
import tensorflow
import mujoco_py
print('loaded mjcpy')
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import fmin_l_bfgs_b
from scipy.signal import savgol_filter
from ou_noise import OU_Process
import torch.nn.functional as F
import foundation as fd
import foundation.util as util
from foundation.envs import Arm
from foundation.control.controllers import MPPI, iLQR
from foundation.control.schemes import OnlineMPC, IterativeMPC
from foundation.control.cost import QuadraticCost
from encoders import Se2NetEncoder

args = None

def run_batch_mpc(env, controller, state_encoder, start_state, time_horizon, action_dim, n_episodes=100):
	mpc = IterativeMPC(controller=controller, time_horizon=time_horizon, action_dim=action_dim)
	mpc.init("random")

	np.set_printoptions(suppress=True)

	res = {}
	res['frames'] = []
	res['actions'] = []
	res['gt_states'] = []

	for ep in range(n_episodes):
		env.set_reset(start_state)
		prev_pos = state_encoder.encode(env.render().copy())
		gtstate, reward, done, _ = env.step(np.array([0]))
		pos = state_encoder.encode(env.render().copy())
		vel = (pos - prev_pos)
		latent_x = torch.cat([pos, vel]).view(-1)  # 2 x 3 x 4
		prev_pos = pos

		actions = []
		frames = []
		gt_states = []
		actions.append(0)
		frames.append(env.render())
		gt_states.append((start_state))

		mpc.optimize_policy(latent_x)
		traj = torch.zeros((time_horizon+1, len(latent_x))).cuda()
		u_seq = torch.zeros((time_horizon, action_dim)).cuda()
		traj[0] = latent_x
		for i in range(time_horizon):
			action = mpc.get_next_command(latent_x, i)
			obs_state, _, _, _ = env.step(action.detach().cpu().numpy().clip(-3,3))
			rgb = env.render()
			pos = state_encoder.encode(rgb.copy())
			vel = (pos - prev_pos)
			latent_x = torch.cat([pos, vel]).view(-1)  # 2 x 3 x 4
			prev_pos = pos
			traj[i+1] = latent_x
			u_seq[i, :] = action
			gt_states.append(obs_state)
			actions.append(action)
			frames.append(rgb)
			if i % 2 == 0:
				print('ep:{}/{}, step:{}/{} state={} action={:.3f}'.format(ep+1, n_episodes, i+1, time_horizon, np.round(obs_state,3), action[0]))

		J = mpc.eval_policy(traj, u_seq)
		print('Eval: {:.2f}'.format(J))

		res['gt_states'].append(gt_states)
		res['actions'].append(actions)
		#res['frames'].append(frames)
	return res

def run_online_mpc(env, controller, state_encoder, start_state, action_dim,
				   optimization_horizon, episode_length, n_episodes=100):

	mpc = OnlineMPC(controller=controller, time_horizon=optimization_horizon,
					action_dim=action_dim, receding_horizon=False)
	mpc.init("zero")

	res = {}
	res['frames'] = []
	res['actions'] = []
	res['gt_states'] = []

	for ep in range(n_episodes):
		actions = []
		frames = []
		gt_states = []

		env.set_reset(start_state)
		prev_pos = state_encoder.encode(env.render().copy())
		gtstate, reward, done, _ = env.step(np.array([0]))
		rgb = env.render()
		pos = state_encoder.encode(rgb.copy())
		vel = (pos - prev_pos)
		latent_x = torch.cat([pos, vel]).view(-1)  # 2 x 3 x 4

		mpc.reset(optimization_horizon, init_type="random")

		actions.append(0)
		frames.append(rgb)
		gt_states.append((start_state))

		traj = torch.zeros((episode_length+1, len(latent_x))).cuda()
		u_seq = torch.zeros((episode_length, action_dim)).cuda()
		traj[0] = latent_x
		for i in range(episode_length):
			print("episode: ", ep, " time step: ", i)

			pos = state_encoder.encode(rgb.copy())
			vel = (pos - prev_pos)
			latent_x = torch.cat([pos, vel]).view(-1)  # 2 x 3 x 4
			prev_pos = pos
			mpc.optimize_policy(latent_x)
			action = mpc.get_next_command()

			obs_state, _, _, _ = env.step(action)

			if i % 2 == 0:
				print('ep:{}/{}, step:{}/{} state={} action={:.3f}'.format(ep + 1, n_episodes, i + 1, episode_length,
																		   np.round(obs_state, 3), action[0]))

			traj[i+1] = latent_x
			u_seq[i, 0] = action
			gt_states.append(obs_state)
			actions.append(action)
			rgb = env.render()
			frames.append(rgb)

		J = mpc.eval_policy(traj, u_seq)
		print('Eval: {:.2f}'.format(J))

		res['gt_states'].append(gt_states)
		res['actions'].append(actions)
		#res['frames'].append(frames)
	return res

def main(argv):
	parser = util.setup_control_options()

	np.set_printoptions(suppress=True)

	global args
	args = parser.parse_args()

	args.save_log = False
	if args.save_log:
		now = time.strftime("%b-%d-%Y-%H%M%S")
		util.create_dir(args.log + '_' + now)
		tblogger = util.TBLogger(args.log + '_' + now)  # Start tensorboard logger
		tblogger.scalar_summary('zzz-ignore', 19., 0)
		print('Printing stats to tensorboard log at {}'.format(args.log + '_' + now))
	else:
		print('Not printing stats to tensorboard')

	if args.seed < 0:
		args.seed = np.random.randint(0,1000000)
		print('Generated new seed:',args.seed)

	if args.encoder == '-':
		args.encoder = None

	# Setup environment
	
	if args.env == 'cartpole':
		
		assert False
		
		from cartpole.mujoco_pendulum_env import Mujoco_Pendulum

		env_args = {'pbc':args.pbc, 'limited':args.limited}
	
		env = Mujoco_Pendulum(**env_args)
		
		args.num_ctrl = 1
		args.num_state = 4
		
		start_state = np.array([0, np.pi, 0, 0])
		target_state = np.array([0, 0, 0, 0])
		
		assert args.encoder is not None
		
	elif args.env == 'arm':
		env = Arm(moving_target=False)
		
		args.num_ctrl = 2
		args.num_state = 2*3
		
		start_set
		
		state = env.reset(x_0=torch.FloatTensor([-np.pi/2, -np.pi/2, 0.9499, np.pi/2]), dx_0=torch.zeros(4), goal=None)
		
	else:
		raise Exception('unknown env: {}'.format(args.env))
	
	if args.horizon <= 0:
		args.horizon = env.horizon

	if args.encoder is not None:
		encoder = Se2NetEncoder(args.encoder) if args.encoder is not None else None

	assert not args.use_gt_model, 'no gt dynamics currently'

	args.dt = 1. # 1/60

	args.def_type = 'torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor'

	dynamics = util.load_transition_model(args.transition)
	assert args.cuda, 'only cuda is supported'
	if args.cuda:
		dynamics.cuda()

	print('Start state: {}\nTarget state: {}'.format(str(start_state.round(3)), str(target_state.round(3))))

	if encoder is not None:
		env.set_reset(target_state)
		prev_pos = encoder.encode(env.render().copy())
		env.step(np.array([0]))
		pos = encoder.encode(env.render().copy())
		vel = (pos - prev_pos)
		latent_target = torch.cat([pos, vel]).view(-1)  # 2 x 3 x 4

	args.num_ctrl = 1
	args.num_state = len(latent_target)

	Q = torch.ones(args.num_state).cuda()

	# cost shaping
	# alpha, beta, gamma = 100, 10, 1  # theta, x, vx
	#
	# Q[8] = alpha
	# Q[9] = alpha
	# Q[10] = alpha
	# Q[11] = alpha
	# Q[0] = beta
	# Q[12] = gamma

	Q = torch.diag(Q)

	Q_terminal = args.alpha * Q.clone()

	R = torch.FloatTensor([[5.]]).cuda()

	# Setup control alg

	if args.alg == 'ilqr':

		cost = QuadraticCost(Q, R, Q_terminal=Q_terminal, x_target=latent_target,
							 state_dim=args.num_state, action_dim=args.num_ctrl)
		controller = iLQR(dynamics, cost, args.dt)

	elif args.alg == 'mppi':

		Q *= args.dt
		cost = QuadraticCost(Q, R*0, Q_terminal=Q_terminal, x_target=latent_target,
							 state_dim=args.num_state, action_dim=args.num_ctrl)
		stoc_proc = OU_Process(dim=(args.num_iter, args.num_ctrl), sigma=args.std, theta=args.theta)
		controller = MPPI(dynamics, cost, args.dt, stoc_proc=stoc_proc, lmbda=1.0,
						  n_rollouts=args.num_iter)

	elif args.alg == 'bp':
		assert False, 'not supported yet'
	elif args.alg == 'lbfgs':
		assert False, 'not supported yet'
	else:
		raise Exception('No control alg')

	print('Using {} control algorithm'.format(args.alg))

	assert args.num_iter > 0 # N
	assert args.seq_len > 0 # K
	print('num-iter={}, seq-len={}'.format(args.num_iter, args.seq_len))

	if args.render:
		assert args.test_len <= 10, 'cant/shouldnt render more than 10 episodes at a time'
		episode_frames = []
		episode_actions = []

	if args.print_freq is None:
		args.print_freq = max(1, args.horizon // 100)

	assert not args.record or args.save_stats, 'cant record frames without saving stats'

	mpc_extra_args = {}

	if args.mpc == 'iterative':
		mpc = IterativeMPC
		args.seq_len = args.horizon
		print('Using iterative MPC: horizon={}'.format(args.seq_len))
	elif args.mpc == 'online':
		mpc = OnlineMPC
		if args.receding_horizon:
			mpc_extra_args['ep_len'] = args.horizon
		print('Using online MPC: rollout-len={}{}'.format(args.seq_len, ', w/ receding horizon' if args.receding_horizon else ''))
	else:
		raise Exception('unknown mpc type: {}'.format(args.mpc))

	mpc = mpc(controller, horizon=args.seq_len, warm_start=args.warm_start, init_type='zero',
			  state_dim=args.num_state, action_dim=args.num_ctrl, **mpc_extra_args)

	for test in range(args.test_len):

		# reset stats for each test
		stats = util.StatsMeter('loss-min', 'loss-max', 'terminal-cost', 'ctrl', 'target', 'timing_s', 'timing_c',
								'error-total', 'error-x', 'error-th', 'error-vx', 'error-vth')

		frames = deque()
		actions = deque()
		if args.save_stats is not None:
			save_stats = ['timing', 'gtstate', 'pose', 'control', 'stats']

			if args.record:
				assert args.test_len <= 10, 'cant/shouldnt render more than 10 episodes at a time'
				save_stats.append('rgb')
				save_stats.append('mask')

			save_stats = {name: deque() for name in save_stats}

		ctrl_seq = torch.zeros(args.horizon, args.num_ctrl).type(args.def_type)
		state_traj = torch.zeros(args.horizon+1, args.num_state).type(args.def_type)

		mpc.reset()

		ep_start = time.time()

		# reset env
		if args.const_initial_state:
			gtstate = env.set_reset(start_state)
		else:
			gtstate = env.reset()
		# get vel estimate by stepping once w/out control
		prev_pos = encoder.encode(env.render().copy())
		gtstate, reward, done, _ = env.step(np.array([0]))

		for step in range(args.horizon):

			rgb = env.render()

			# estimate state
			start = time.time()
			pos = encoder.encode(rgb.copy())
			vel = (pos - prev_pos)
			state = torch.cat([pos, vel]).view(-1)  # 2 x 3 x 4
			prev_pos = pos
			stats.update('timing_s', time.time() - start)

			state_traj[step] = state

			# compute control
			start = time.time()
			ctrl = mpc.get_next_command(state, step).clamp(-3,3)
			stats.update('timing_c', time.time() - start)

			ctrl_seq[step] = ctrl

			if args.render or args.record:
				frames.append(rgb)
				if args.render:
					actions.append(ctrl[0].item())

			if args.save_stats is not None:
				save_stats['gtstate'].append(gtstate)

				if args.record:
					save_stats['mask'].append(encoder.encode(rgb.copy(), get_masks=True)[1][0].permute(1,2,0).cpu().numpy())

			# take 1 step
			gtstate, _, done, _ = env.step(ctrl.detach().cpu().numpy())

			err = (gtstate - target_state)**2
			err[1] = util.angle_diff_scalar(gtstate[1], target_state[1])**2

			stats.update('error-x', np.sqrt(err[0]))
			stats.update('error-th', np.sqrt(err[1]))
			stats.update('error-vx', np.sqrt(err[2]))
			stats.update('error-vth', np.sqrt(err[3]))

			stats.update('error-total', np.sqrt(err.sum()) / 4)
			stats.update('ctrl', np.sqrt((ctrl ** 2).sum()))

			if step % args.print_freq == 0:
				print('Test {}/{} Step {}/{}: state={}, loss={:.4f} (max={:.4f}), end-cost={:.4f}, action={:.3f}'.format(test+1, args.test_len, step+1, args.horizon,
										str(state.round(3)) if args.encoder is None else str(gtstate.round(3)),
										stats['loss-min'].val, stats['loss-max'].val, stats['terminal-cost'].val, ctrl[0]))

			if args.save_log:
				info = {name: stats[name].val for name in stats.keys()}
				for tag, value in info.items():
					tblogger.scalar_summary(tag, value, test*args.horizon+step)

		frames.append(env.render())
		pos = encoder.encode(frames[-1].copy())
		vel = (pos - prev_pos)
		state_traj[-1] = torch.cat([pos, vel]).view(-1)

		episode_loss = mpc.eval_trajectory(state_traj, ctrl_seq)
		print('Episode {}/{} loss: {}'.format(test+1, args.test_len, episode_loss))

		if args.render:
			episode_frames.append(frames)
			episode_actions.append(actions)

		if args.save_stats is not None:
			# include final state

			save_stats['loss'] = episode_loss

			save_stats['timing'] = (time.time() - ep_start)/args.horizon
			save_stats['stats'] = stats.export()
			save_stats['args'] = args

			save_stats['gtstate'] = np.array(save_stats['gtstate'])
			save_stats['pose'] = state_traj.detach().cpu().numpy()
			save_stats['control'] = ctrl_seq.detach().cpu().numpy()

			if args.record:
				save_stats['mask'].append(encoder.encode(rgb.copy(), get_masks=True)[1][0].permute(1,2,0).cpu().numpy())
				save_stats['mask'] = np.array(save_stats['mask'])
				save_stats['rgb'] = np.array(frames)
			# save stats
			fname = args.save_stats + '_{}.pth.tar'.format(test)
			torch.save(save_stats, fname)

			print('-- test {} stats saved to {} --'.format(test+1, fname))

	print('Stats for last episode:')
	for name, stat in stats.items():
		if stat.count > 1:
			print('{}: {stat.avg:.4f} +/- {stat.std:.4f} (min={stat.min:.4f},max={stat.max:.4f})'.format(name, stat=stat))

	if args.render:
		print('rendering {} episodes'.format(args.test_len))

		fig, ax = plt.subplots()
		plt.ion()

		for ep, (frames, actions) in enumerate(zip(episode_frames, episode_actions)):

			if args.record is not None:
				imgs = []

			for step, (frame, ctrl) in enumerate(zip(frames, actions)):

				ax.set_title('Episode= {} - {}'.format(ep,step))
				ax.set_xlabel('u = {:.4f}'.format(ctrl))
				ax.set_xticks([])
				ax.set_yticks([])
				ax.imshow(frame)
				fig.tight_layout()
				plt.pause(0.01)

				# if args.record is not None and step % 2 == 0:
				# 	w, h = fig.canvas.get_width_height()
				# 	buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(w,h,4)[::2,::2,1:]
				# 	print(buf.dtype, buf.min(), buf.max())
				# 	print(buf.shape)
				# 	imgs.append(buf)

				plt.cla()




	print('Done.')

if __name__ == '__main__':
	sys.exit(main(sys.argv))
