
import os
import torch

from .. import envs
from .. import agents
from ..models import networks as nets
from foundation.foundation.rl import baselines as baselines, policies as policies

print('WARNING: "setup" is deprecated')




























def load_optimizer(optim_type, parameters, lr=1e-3, momentum=0.9, weight_decay=1e-4):
	if optim_type == 'sgd':
		optimizer = torch.optim.SGD(params=parameters, lr=lr, momentum=momentum,
									weight_decay=weight_decay)
	elif optim_type == 'adam':
		optimizer = torch.optim.Adam(params=parameters, lr = lr, weight_decay= weight_decay)
	elif optim_type == 'rmsprop':
		optimizer = torch.optim.RMSprop(params=parameters, lr=lr, momentum=momentum,
										weight_decay=weight_decay)
	else:
		raise Exception("Unknown optimizer type: " + optim_type)

	return optimizer

def get_agent(agent_name, spec, args, policy=None, baseline=None):

	if agent_name is None or agent_name == '':
		return None

	if agent_name == 'walking':

		sub_model = nets.MLP(4+args.c2s_channels, 2+args.s2c_channels*2, hidden_dims=args.hidden, nonlinearity=args.nonlinearity)

		sub_policy = policies.Normal_MultiCat_Policy(sub_model, num_normal=2, def_type=args.def_type, num_var=args.s2c_channels,
		                                             min_log_std=args.min_log_std, init_log_std=-1.2, )
		
		sub_baseline = baselines.LinearBaselineLeastSqrs(spec[1].obs_space.size, time_order=args.b_time_order, obs_order=args.b_obs_order)
		
		print('Subagent Model: {}'.format(sub_model))

		cmd_policy = None
		cmd_baseline = None
		if args.commander:

			cmd_model = nets.MLP(4*args.s2c_channels, 2*4*args.c2s_channels,
								 hidden_dims=args.cmd_hidden, nonlinearity=args.nonlinearity)
			cmd_policy = policies.MultiCat_Policy(model=cmd_model, num_vars=4 * args.c2s_channels, def_type=args.def_type)
			
			cmd_baseline = baselines.LinearBaselineLeastSqrs(spec[0].obs_space.size, time_order=args.b_time_order, obs_order=args.b_obs_order)
			
			print('Commander Model: {}'.format(cmd_model))

		agent = agents.Multi_NPG(sub_policy=sub_policy, num_sub=4, cmd_policy=cmd_policy, sub_baseline=sub_baseline, cmd_baseline=cmd_baseline,
								 parallel=args.parallel_update, unique_subs=args.unique_subs, separate_stats=args.separate_stats,
								 discount=args.discount, step_size=args.step_size, max_cg_iter=10, reg_coeff=1e-3)
		
	elif agent_name == 'walking-global-baseline':
		
		sub_model = nets.MLP(4 + args.c2s_channels, 2 + args.s2c_channels * 2, hidden_dims=args.hidden,
		                     nonlinearity=args.nonlinearity)
		
		sub_policy = policies.Normal_MultiCat_Policy(sub_model, num_normal=2, def_type=args.def_type,
		                                             num_var=args.s2c_channels, min_log_std=args.min_log_std, init_log_std=0, )
		
		print('Subagent Model: {}'.format(sub_model))
		
		cmd_policy = None
		if args.commander:
			cmd_model = nets.MLP(spec[0].obs_space.size, 2*4*args.c2s_channels,
			                     hidden_dims=args.cmd_hidden, nonlinearity=args.nonlinearity)
			cmd_policy = policies.MultiCat_Policy(model=cmd_model, num_vars=4 * args.c2s_channels,
			                                      def_type=args.def_type)
			
			print('Commander Model: {}'.format(cmd_model))
			
		global_baseline = baselines.LinearBaselineLeastSqrs(sum([s.obs_space.size for s in spec]), time_order=args.b_time_order,
		                                                    obs_order=args.b_obs_order)

		agent = agents.Global_Baseline_NPG(global_baseline=global_baseline, sub_policy=sub_policy, num_sub=4, cmd_policy=cmd_policy,
		                         parallel=args.parallel_update, unique_subs=args.unique_subs,
		                         separate_stats=args.separate_stats,
		                         discount=args.discount, step_size=args.step_size, max_cg_iter=10, reg_coeff=1e-3)

	elif agent_name == 'walking-contin-comm':
		sub_model = nets.MLP(spec[1].obs_space.size, spec[1].action_space.size, hidden_dims=args.hidden,
							 nonlinearity=args.nonlinearity)

		sub_policy = policies.Gaussian_Policy(sub_model, def_type=args.def_type,
		                                      min_log_std=args.min_log_std,
		                                      init_log_std=0, )

		print('Subagent Model: {}'.format(sub_model))

		cmd_policy = None
		if args.commander:
			cmd_model = nets.MLP(spec[0].obs_space.size, spec[0].action_space.size,
								 hidden_dims=args.cmd_hidden, nonlinearity=args.nonlinearity)
			cmd_policy = policies.Gaussian_Policy(model=cmd_model, def_type=args.def_type, min_log_std=args.min_log_std,
			                                      init_log_std=0, )

			print('Commander Model: {}'.format(cmd_model))

		global_baseline = baselines.LinearBaselineLeastSqrs(sum([s.obs_space.size for s in spec]),
		                                                    time_order=args.b_time_order,
		                                                    obs_order=args.b_obs_order)

		agent = agents.Global_Baseline_NPG(global_baseline=global_baseline, sub_policy=sub_policy, num_sub=4,
										   cmd_policy=cmd_policy,
										   parallel=args.parallel_update, unique_subs=args.unique_subs,
										   separate_stats=args.separate_stats,
										   discount=args.discount, step_size=args.step_size, max_cg_iter=10,
										   reg_coeff=1e-3)

	elif agent_name == 'walking-simple':

		assert False

		# TODO: test completely separated model - Branched net
		sub_model = nets.MLP(4 + args.c2s_channels, 2 + args.s2c_channels * 2, hidden_dims=args.hidden,
							 nonlinearity=args.nonlinearity)

		sub_policy = policies.Gaussian_Policy(sub_model, def_type=args.def_type,
		                                      min_log_std=args.min_log_std, init_log_std=-1.5, )

		sub_baseline = baselines.LinearBaselineLeastSqrs(spec[1].obs_space.size, time_order=args.b_time_order,
		                                                 obs_order=args.b_obs_order)

		print('Subagent Model: {}'.format(sub_model))

		cmd_policy = None
		cmd_baseline = None
		if args.commander:
			cmd_model = nets.MLP(spec[0].obs_space.size, spec[0].act_space.size,
								 hidden_dims=args.cmd_hidden, nonlinearity=args.nonlinearity)
			cmd_policy = policies.Gaussian_Policy(model=cmd_model, min_log_std=args.min_log_std, init_log_std=-1.5, def_type='torch.FloatTensor')

			cmd_baseline = baselines.LinearBaselineLeastSqrs(spec[0].obs_space.size, time_order=args.b_time_order,
			                                                 obs_order=args.b_obs_order)

			print('Commander Model: {}'.format(cmd_model))

		agent = agents.Multi_NPG(sub_policy=sub_policy, num_sub=4, cmd_policy=cmd_policy, sub_baseline=sub_baseline,
								 cmd_baseline=cmd_baseline,
								 parallel=args.parallel_update, unique_subs=args.unique_subs,
								 separate_stats=args.separate_stats,
								 discount=args.discount, step_size=args.step_size, max_cg_iter=10, reg_coeff=1e-3)

	else:
		raise Exception('Unknown agent name: {}'.format(agent_name))

	return agent # or manager

def get_env_type(env_name, args=[], kwargs={}):
	# init env
	if env_name == 'ball':
		args = ['DRL_PointMass-v0']
		kwargs = {}
		return envs.make_env_wrapper(envs.GymEnv), args, kwargs
	elif env_name == 'ant':
		args = ['DRL_Ant-v0']
		kwargs = {}
		return envs.make_env_wrapper(envs.GymEnv), args, kwargs
	elif env_name == 'cheetah':
		args = ['DRL_HalfCheetah-v0']
		kwargs = {}
		return envs.make_env_wrapper(envs.GymEnv), args, kwargs
	elif env_name == 'swimmer':
		args = ['DRL_Swimmer-v0']
		kwargs = {}
		return envs.make_env_wrapper(envs.GymEnv), args, kwargs
	elif env_name == 'pendulum':
		args = ['Pendulum-v0']
		kwargs = {}
		return envs.make_env_wrapper(envs.GymEnv), args, kwargs
	elif env_name == 'cartpole':
		args = ['CartPole-v1']
		kwargs = {}
		return envs.make_env_wrapper(envs.GymEnv), args, kwargs
	elif env_name == 'simple-ant':
		args = ['SimpleAnt-v0']
		kwargs = {}
		return envs.make_env_wrapper(envs.GymEnv), args, kwargs
	elif env_name == 'sum':
		raise Exception('update args in util.setup.get_env_type')
		kwargs = {k:v for k, v in kwargs.items() if k in {'!!!missing!!!'}}
		return envs.Sum_Env, args, kwargs
	elif env_name == 'sel':
		raise Exception('update args in util.setup.get_env_type')
		kwargs = {k: v for k, v in kwargs.items() if k in {'!!!missing!!!'}}
		return envs.Sel_Env, args, kwargs
	elif env_name == 'arm':
		raise Exception('update args in util.setup.get_env_type')
		kwargs = {k: v for k, v in kwargs.items() if k in {'!!!missing!!!'}}
		return envs.Arm, args, kwargs
	elif env_name == 'mnist':
		kwargs = {k: v for k, v in kwargs.items() if k in {'batch_size', 'traindata', 'download', 'stochastic', 'sparse_reward', 'random_goal', 'loop'}}
		return envs.MNIST_Walker, args, kwargs
	elif env_name == 'walking':
		raise Exception('update args in util.setup.get_env_type')
		kwargs = {k: v for k, v in kwargs.items() if k in {'!!!missing!!!'}}
		return envs.Walking, args, kwargs
	else:
		raise Exception('unknown env type: {}'.format(env_name))

def get_env(env_name, args=[], kwargs={}):

	env_type, args, kwargs = get_env_type(env_name, args=args, kwargs=kwargs)
	return env_type(*args, **kwargs)

def get_model(model_type):
	# setup model
	if model_type == 'mlp':
		return nets.MLP#(obs_dim, action_dim, hidden_dims=args.hidden, nonlinearity=args.nonlinearity)
	else:
		raise Exception('Unknown model name: {}'.format(model_type))

def get_policy(policy_type):
	# setup policy
	if policy_type == 'gaussian':
		return policies.Gaussian_Policy#(args.model, min_log_std=args.min_log_std, init_log_std=0, def_type=args.def_type)
	elif policy_type == 'cat':
		return policies.Categorical_Policy#(args.model, def_type=args.def_type)
	else:
		raise Exception('Unknown policy name: {}'.format(policy_type))

def get_baseline(baseline_type):
	# setup baseline
	if baseline_type == 'lin':
		return baselines.LinearBaselineLeastSqrs#(obs_dim, time_order=args.b_time_order, obs_order=args.b_obs_order)
	elif baseline_type == 'mlp':
		return baselines.MLP_Baseline
	else:
		raise Exception('unknown baseline: {}'.format(baseline_type))

def get_alg(alg_type):
	if alg_type == 'npg':
		return agents.NPG
	elif alg_type == 'vpg':
		return agents.VPG
	else:
		raise Exception('unknown algorithm: {}'.format(alg_type))

def setup_multi_system(args):
	
	env = get_env(args.env, args)
	
	assert args.agent is not None, 'Multi-Agent currently requires a predefined (in the code) agent'
	
	agent = get_agent(args.agent, env.spec, args)
	return agent, env

def setup_system(args):

	env = get_env(args.env, args)

	# manually set up agent - only for general envs

	obs_dim = env.spec.obs_space.size
	action_dim = env.spec.action_space.size if isinstance(env.spec.action_space, envs.Continuous_Space) else env.spec.action_space.choices

	# setup model
	model_type = get_model(args.model)
	if args.model == 'mlp':
		model = model_type(obs_dim, action_dim, hidden_dims=args.hidden, nonlinearity=args.nonlinearity)

	# setup policy
	policy_type = get_policy(args.policy)
	if args.policy == 'gaussian':
		policy = policy_type(model, min_log_std=args.min_log_std, init_log_std=0, def_type=args.def_type)
	elif args.policy == 'cat':
		policy = policy_type(model, def_type=args.def_type)

	# setup baseline
	if args.baseline == 'lin':
		baseline = baselines.LinearBaselineLeastSqrs(obs_dim, time_order=args.b_time_order, obs_order=args.b_obs_order)
	elif args.baseline == 'mlp':
		baseline = baselines.MLP_Baseline(obs_dim, hidden_dims=args.b_hidden, epochs=args.b_epochs, batch_size=args.b_batch_size,
		                                  optim_type=args.b_optim_type, lr=args.b_lr, weight_decay=args.b_weight_decay,
		                                  momentum=args.b_momentum, nesterov=args.b_nesterov)

	if args.agent is not None:
		agent = get_agent(args.agent, env.spec, args, policy=policy, baseline=baseline)
		return agent, env

	# setup agent (alg)

	alg_type = get_alg(args.alg)
	if args.alg == 'vpg':
		agent = alg_type(policy, baseline=baseline, discount=args.discount, optim_type=args.optim_type, lr=args.lr,
						   weight_decay=args.weight_decay, momentum=args.momentum)
	elif args.alg == 'npg':
		agent = alg_type(policy, baseline=baseline, discount=args.discount, step_size=args.step_size, max_cg_iter=10,
						   reg_coeff=1e-3)

	return agent, env

def check_load_args(args, load_args):
	#assert load_args.agent == args.agent
	#assert load_args.env == args.env
	#assert load_args.policy == args.policy
	return load_args

def save_checkpoint(args, agent, logger, episode, best, is_best=False):
	assert episode or is_best
	name = 'model_best.pth.tar' if is_best else 'checkpoint_{}.pth.tar'.format(episode)
	save_path = os.path.join(args.save_dir, name)
	ckpt = {
		'agent': agent.state_dict(),
		'args': args,
		'total-episodes': episode,
		'best': best,
	}
	torch.save(ckpt, save_path)
	if logger is not None:
		logger.save(os.path.join(args.save_dir, 'stats.pth.tar'))
	return save_path

def load_checkpoint(path): # either from job name or specific checkpoint
	if os.path.isdir(path):
		latest = os.listdir(path)[-1]
		if 'pth.tar' in latest:
			path = os.path.join(path, 'model_best.pth.tar')
		else:
			path = os.path.join(path, latest, 'model_best.pth.tar')

	return path, torch.load(path)

