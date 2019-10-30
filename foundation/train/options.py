import configargparse

def get_parser(desc='blank', no_config=False):
	parser = configargparse.ArgumentParser(description=desc)

	if not no_config:
		parser.add_argument('-c', '--config', required=True, is_config_file=True,
		                    help='Path to config file for parameters')
	return parser

def setup_standard_options(parser=None, no_config=False):
	if parser is None:
		parser = get_parser(no_config=no_config)

	# Saving
	parser.add_argument('-s', '--saveroot', type=str, default='../trained_nets/', )
	parser.add_argument('-n', '--name', type=str, default=None)
	parser.add_argument('--no-tb', dest='tblog', action='store_false')
	parser.add_argument('--txtlog', action='store_true')
	parser.add_argument('--logdate', action='store_true')
	parser.add_argument('--print-freq', type=int, default=-1)
	parser.add_argument('--save-freq', type=int, default=-1)
	parser.add_argument('--track-best', action='store_true')

	parser.add_argument('--resume', type=str, default=None)
	parser.add_argument('-l', '--load', type=str, default=None)

	# Training
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=25)
	parser.add_argument('--start-epoch', type=int, default=0)
	parser.add_argument('--decay-epochs', type=int, default=-1)
	parser.add_argument('--decay-factor', type=float, default=0.2)
	parser.add_argument('--use-val', action='store_true')
	parser.add_argument('--test-per', type=float, default=None)
	parser.add_argument('--val-per', type=float, default=None)
	parser.add_argument('--no-unique-tests', dest='unique_tests', action='store_false')
	parser.add_argument('-b', '--batch-size', default=128, type=int, )
	parser.add_argument('--no-test', action='store_true')
	parser.add_argument('--viz-criterion-args', nargs='+', type=str, default=None)
	parser.add_argument('--stats-decay', type=float, default=.01)

	# Optim
	parser.add_argument('--optim-type', type=str, default='adam')
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--weight-decay', type=float, default=0)
	parser.add_argument('--momentum', type=float, default=0)

	# Device/Multiprocessing
	parser.add_argument('-j', '--num-workers', type=int, default=4)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--no-cuda', action='store_true')

	# Dataset
	parser.add_argument('--dataroot', type=str, default=None) # a common root
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('-d', '--data', type=str, nargs='+')
	parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
	parser.add_argument('--drop-last', action='store_true')
	parser.add_argument('--indexed', action='store_true')

	# Model
	parser.add_argument('--model-type', type=str, default=None)
	parser.add_argument('--latent-dim', type=int, default=-1)
	parser.add_argument('--nonlin', type=str, default='prelu')
	parser.add_argument('--fc', type=int, nargs='+', default=None)
	parser.add_argument('--channels', type=int, nargs='+', default=None)
	parser.add_argument('--kernels', type=int, nargs='+', default=None)
	parser.add_argument('--factors', type=int, nargs='+', default=None)
	parser.add_argument('--strides', type=int, nargs='+', default=1)
	parser.add_argument('--upsampling', type=str, default='bilinear')
	parser.add_argument('--downsampling', type=str, default='max')
	parser.add_argument('--norm-type', type=str, default='instance')

	return parser





##########
# Old



def setup_unsup_options():
	parser = configargparse.ArgumentParser(description='Generate data of')

	parser.add_argument('-c', '--config', required=True, is_config_file=True,
						help='Path to config file for parameters')


	# Saving
	parser.add_argument('-s', '--saveroot', type=str, default='trained_nets/',)
	parser.add_argument('-n', '--name', type=str, default='test')
	parser.add_argument('--no-tb', dest='tblog', action='store_false')
	parser.add_argument('--txtlog', action='store_true')
	parser.add_argument('--logdate', action='store_true')
	parser.add_argument('--print-freq', type=int, default=-1)

	parser.add_argument('--resume', type=str, default=None)
	parser.add_argument('-l', '--load', type=str, default=None)

	# Training
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=25)
	parser.add_argument('--start-epoch', type=int, default=0)
	parser.add_argument('--decay-epochs', type=int, default=12)
	parser.add_argument('--decay-factor', type=float, default=0.1)
	parser.add_argument('--test-per', default=0.1, type=float,)
	parser.add_argument('--val-per', default=0.1, type=float, )
	parser.add_argument('--use-val', action='store_true' )
	parser.add_argument('--no-unique-tests', dest='unique_tests', action='store_false')
	parser.add_argument('-b', '--batch-size', default=32, type=int,)
	parser.add_argument('--no-test', action='store_true')

	# Optim
	parser.add_argument('--optim-type', type=str, default='adam')
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--weight-decay', type=float, default=1e-4)
	parser.add_argument('--momentum', type=float, default=0.9)

	# Device/Multiprocessing
	parser.add_argument('-j', '--num-workers', type=int, default=4)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--no-cuda', action='store_true')

	# Dataset
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('-d','--data', type=str, nargs='+')

	# Model
	parser.add_argument('--din', type=int, nargs='+', default=None)
	parser.add_argument('--model-type', type=str, default='auto')
	parser.add_argument('--nonlin', type=str, default='prelu')
	parser.add_argument('--latent-dim', type=int, default=8)
	parser.add_argument('--fc', type=int, nargs='+', default=[32])

	parser.add_argument('--channels', type=int, nargs='+', default=[8])
	parser.add_argument('--kernels', type=int, nargs='+', default=3)
	parser.add_argument('--factors', type=int, nargs='+', default=2)
	parser.add_argument('--downsampling', type=str, default='max')
	parser.add_argument('--upsampling', type=str, default='deconv')
	parser.add_argument('--batch-norm', action='store_true')

	parser.add_argument('--beta', type=float, default=1) # for beta-VAE

	# Loss
	parser.add_argument('--loss-type', type=str, default='mse')
	parser.add_argument('--loss-scale', type=float, default=1.)


	return parser



def setup_rl_options():
	# Parse arguments
	parser = configargparse.ArgumentParser(description='Policy Gradient Training')

	# Dataset options
	parser.add_argument('-c', '--config', required=False, is_config_file=True,
	                    help='Path to config file for parameters')

	parser.add_argument('--name', type=str, default=None, help='Name of job (required if not resuming)')

	parser.add_argument('--save-root', type=str, default='', help='Root dir for results')
	parser.add_argument('--log-date', action='store_true', help='Log stats on Tensorboard')
	parser.add_argument('--log-tb', action='store_true', help='Log stats on Tensorboard')
	parser.add_argument('--log-txt', action='store_true', help='Log stats on Tensorboard')
	parser.add_argument('--save-freq', type=int, default=None, help='Total training iterations')

	parser.add_argument('--agent', type=str, default=None, help='Agent Name (for predefined agents)')
	parser.add_argument('--clip', type=float, default=0.3, help='Learning rate')

	parser.add_argument('--policy', type=str, default='mlp', help='Policy Architecture name')
	parser.add_argument('--model', type=str, default='mlp', help='Policy Architecture name')
	parser.add_argument('--baseline', type=str, default='lin', help='Baseline name')

	parser.add_argument('--env', type=str, default='ball', help='Environment name')

	# parser.add_argument('--resume', type=str, default=None, help='Path to resume')
	# parser.add_argument('--load', type=str, default=None, help='Load previous run and begin new')

	parser.add_argument('--device', type=str, default='cpu', help='Use CUDA (only allows 1 worker on windows)')
	parser.add_argument('--seed', type=int, default=None, help='Random Seed')
	parser.add_argument('--budget-steps', type=int, default=1e6)
	parser.add_argument('--steps-per-itr', type=int, default=2048)
	parser.add_argument('--tau', type=float, default=0.1)

	parser.add_argument('--epochs', type=int, default=10, help='Epochs per train step for MLP baseline')
	parser.add_argument('--batch-size', type=int, default=128,
	                    help='Batch size per epoch in train step for MLP baseline')

	parser.add_argument('--norm-adv', action='store_true', help='Log stats on Tensorboard')

	parser.add_argument('--optim-type', type=str, default='rmsprop', help='Default Optimizer')
	parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 Weight decay for optim')
	parser.add_argument('--momentum', type=float, default=0, help='Momentum for optim')
	parser.add_argument('--step-size', type=float, default=1e-3, help='Line Search max divergence')
	parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
	parser.add_argument('--subsample', type=float, default=0, help='Subsample percentage')
	parser.add_argument('--gae-lambda', type=float, default=0.97, help='GAE Lambda parameter')
	parser.add_argument('--nonlin', type=str, default='elu', help='Network nonlinearity')
	parser.add_argument('--hidden', type=int, nargs='+', default=[], help='Hidden layers for policy')
	parser.add_argument('--min-log-std', type=float, default=None, help='Min log std for gaussian policy')

	parser.add_argument('--b-hidden', type=int, default=[], nargs='+', help='Hidden layers for baseline')
	parser.add_argument('--b-scale-max', action='store_true')

	parser.add_argument('--b-epochs', type=int, default=10, help='Epochs per train step for MLP baseline')
	parser.add_argument('--b-batch-size', type=int, default=128,
	                    help='Batch size per epoch in train step for MLP baseline')
	parser.add_argument('--b-nonlin', type=str, default='elu', help='Nonlinearity for MLP baseline')
	parser.add_argument('--b-optim-type', type=str, default='rmsprop', help='Default Optimizer')
	parser.add_argument('--b-lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--b-weight-decay', type=float, default=1e-4, help='L2 Weight decay for optim')
	parser.add_argument('--b-momentum', type=float, default=0, help='Momentum for optim')
	parser.add_argument('--b-nesterov', action='store_true', help='Use nesterov accelerated grad (only with SGD)')

	parser.add_argument('--b-time-order', type=int, default=3,
	                    help='Powers of timestep index to use as features for baseline')
	parser.add_argument('--b-obs-order', type=int, default=1,
	                    help='Powers of observations to use as features for baseline')

	return parser




# OLD

def setup_gen_data_options():
	parser = configargparse.ArgumentParser(description='Generate data of')

	parser.add_argument('--num', type=int, default=10, help='Number of sequences to generate for each set')
	parser.add_argument('--sets', type=int, default=1, help='Number of sets to generate')
	parser.add_argument('--set-offset', type=int, default=0, help='Offset for dataset naming')
	parser.add_argument('-s', '--save', default='results', type=str, metavar='PATH',
	                    help='path to save results in (w/out suffix)')

	parser.add_argument('--seq-len', default=2, type=int,
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

	# parser.add_argument('--limited', action='store_true', help='use limited cartpole')
	# parser.add_argument('--pbc', action='store_true', help='use cartpole with pbc')

	# parser.add_argument('--zero-cart', action='store_true', help='when generating move only 1 joint')
	# parser.add_argument('--separate-joints', action='store_true', help='when generating move only 1 joint')

	return parser


def setup_control_options():
	parser = configargparse.ArgumentParser(description='RL Training/Evaluation')

	parser.add_argument('-c', '--config', required=True, is_config_file=True,
	                    help='Path to config file for parameters')

	parser.add_argument('--print-freq', type=int, default=None)
	parser.add_argument('--filter', action='store_true', )
	parser.add_argument('--use-ou', action='store_true', )
	parser.add_argument('--limited', action='store_true', help='use limited cartpole')
	parser.add_argument('--pbc', action='store_true', help='use cartpole with pbc')
	parser.add_argument('--mpc', type=str, default='online')
	parser.add_argument('--alg', required=True, help='Control algorithm to use')
	parser.add_argument('--transition', type=str, default='-',
	                    help='path to load file (pth.tar containing args, \'-\' for none)')
	parser.add_argument('--encoder', type=str, default='-',
	                    help='path to load file (pth.tar containing args, \'-\' for none)')
	parser.add_argument('--log', type=str, default=None,
	                    help='path to load file (pth.tar containing args, \'-\' for none)')
	parser.add_argument('--encoder-se3', type=str, default='quat',
	                    help='Type of se3 the encoder outputs {quat, aa, rt}')
	parser.add_argument('--encoder-objects', default=[-1], nargs='+', type=int,
	                    help='Indices of poses to be used by encoder (to remove background poses) (negative index to use all poses)')
	parser.add_argument('--render', action='store_true', help='Save episodes that are visualized')

	parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Don\'t use cuda')
	parser.add_argument('--seed', type=int, default=5, help='Seed for random state')
	parser.add_argument('--num-iter', type=int, default=20,
	                    help='Number of training iterations to run now (required)')
	parser.add_argument('--seq-len', type=int, default=20,
	                    help='Number of training iterations to run now (required)')
	parser.add_argument('--test-len', type=int, default=10, help='Number of episodes to test (default 10)')
	parser.add_argument('--no-warm-start', dest='warm_start', action='store_false', help='reset plan after every step')
	parser.add_argument('--pred-next', action='store_true', help='predict next state and print out results')
	parser.add_argument('--use-gt-model', action='store_true', help='use non-differeniable env as transition model')
	parser.add_argument('--penalize-all-states', action='store_true', help='penalize states in entire trajectory')
	parser.add_argument('--const-initial-state', action='store_true', help='set initial state instead of random')
	parser.add_argument('--plan-full', action='store_true', help='Always plan a full sequence')
	parser.add_argument('--plot-poses', action='store_true', help='Always plan a full sequence')
	parser.add_argument('--receding-horizon', action='store_true', help='Always plan a full sequence')
	parser.add_argument('--horizon', type=int, default=0,
	                    help='Number of steps per episode')

	parser.add_argument('--cost-weights', type=float, nargs='+', default=[1, 1, 1, 1], help='what weights to use')
	parser.add_argument('--lmbda', type=float, default=0,
	                    help='If >0 uses levenberg marquadt trick in matrix inversion')
	parser.add_argument('--alpha', type=float, default=1e-4, help='Discount factor')
	parser.add_argument('--std', type=float, default=1e-4, help='Standard deviation of noise for MPPI')
	parser.add_argument('--theta', type=float, default=1e-4, help='OU process param for noise for MPPI')
	parser.add_argument('--ctrl-cost', type=float, default=1e-2, help='Model Parameter')
	parser.add_argument('--loss-scale', type=float, default=1, help='scale of loss')

	parser.add_argument('--record', action='store_true', help='record a video of the episode')
	parser.add_argument('--save-stats', type=str, default=None, help='save results including states and controls')

	return parser


def setup_ray_options():
	parser = configargparse.ArgumentParser(description='Policy Gradient Training')

	# Dataset options
	parser.add_argument('-c', '--config', required=False, is_config_file=True,
	                    help='Path to config file for parameters')

	parser.add_argument('--name', type=str, default=None, help='Name of job (required if not resuming)')

	parser.add_argument('--policy', type=str, default='mlp', help='Policy Architecture name')

	parser.add_argument('--env', type=str, default='ball', help='Environment name')

	parser.add_argument('--resume', type=str, default=None, help='Path to resume')
	parser.add_argument('--load', type=str, default=None, help='Load previous run and begin new')

	parser.add_argument('--save-root', type=str, default='', help='Root dir for results')
	parser.add_argument('--num-workers', type=int, default=8, help='Number of workers')
	parser.add_argument('--cuda', action='store_true', help='Use CUDA (only allows 1 worker on windows)')
	parser.add_argument('--seed', type=int, default=10, help='Random Seed')
	parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
	parser.add_argument('--log', action='store_true', help='Log stats on Tensorboard')

	parser.add_argument('--num-iter', type=int, default=0, help='Total training iterations')
	parser.add_argument('--num-traj', type=int, default=50, help='Number of trajectories per iteration')
	parser.add_argument('--save-freq', type=int, default=5, help='Saving frequency')
	parser.add_argument('--num-eval', type=int, default=None, help='Number of evaluation episodes')

	parser.add_argument('--delta', type=float, default=1e-3, help='Line Search max divergence')
	parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
	parser.add_argument('--nonlinearity', type=str, default='elu', help='Network nonlinearity')
	parser.add_argument('--hidden', type=int, nargs='+', default=[], help='Hidden layers for policy')
	parser.add_argument('--min-log-std', type=float, default=None, help='Min log std for gaussian policy')
	parser.add_argument('--reg-coeff', type=float, default=1e-6, help='Min log std for gaussian policy')
	parser.add_argument('--max-steps', type=int, default=10,
	                    help='Powers of timestep index to use as features for baseline')

	parser.add_argument('--time-order', type=int, default=3,
	                    help='Powers of timestep index to use as features for baseline')
	parser.add_argument('--obs-order', type=int, default=1,
	                    help='Powers of observations to use as features for baseline')

	return parser


def setup_multi_options():
	parser = setup_pg_options()  # include all the same options as for pg training

	parser.add_argument('--manager', type=str, default='master-slaves', help='Type of manager to use for training')
	parser.add_argument('--parallel-update', action='store_true', help='Update each agent in parallel')
	parser.add_argument('--separate-stats', action='store_true', help='Save separate stats for each agent')
	parser.add_argument('--multi-agent', default=True)

	# comms

	parser.add_argument('--continuous-comm', action='store_true')
	parser.add_argument('--s2c-channels', type=int, default=3, help='')
	parser.add_argument('--c2s-channels', type=int, default=3, help='')
	parser.add_argument('--cut-comm-graph', action='store_true', help='')

	parser.add_argument('--commander', action='store_true')
	parser.add_argument('--cmd-hidden', type=int, nargs='+', default=[], help='')
	parser.add_argument('--unique-subs', action='store_true', help='')

	return parser


def setup_viz_options():
	parser = configargparse.ArgumentParser(description='Policy Visualization')

	parser.add_argument('path', type=str, default=None, help='Path to resume')
	parser.add_argument('-n', type=int, default=1, help='Number of episodes to visualize')

	return parser


def setup_q_options():  # TODO: make consistent with pg
	# Parse arguments
	parser = configargparse.ArgumentParser(description='Swarm Ions Training')

	parser.add_argument('-c', '--config', required=True, is_config_file=True,
	                    help='Path to config file for parameters')

	parser.add_argument('--name', required=True, type=str,
	                    help='Environment for agent to learn')
	parser.add_argument('--env', required=True,
	                    help='Environment for agent to learn')
	parser.add_argument('--alg', required=True, choices=['ddpg'],
	                    help='Agent to learn')
	parser.add_argument('--policy', default=None, choices=['branched', 'separate'],
	                    help='Policy type for agent use')
	# parser.add_argument('--baseline', default=None, choices=['linear', 'linear-features', 'MLP', 'zero', 'quadratic'],
	#					help='Baseline for agent')
	parser.add_argument('--save-root', type=str, default=None, help='root to save output')
	parser.add_argument('--resume', type=str, default=None, help='path to resume RL model')
	parser.add_argument('--viz', type=int, default=0, help='Episodes to visualize after training (0 for none)')

	parser.add_argument('--loss-type', type=str, default='mse', help='type of loss function to use')
	parser.add_argument('--nonlinearity', type=str, default='elu', help='nonlinearity to be used')
	parser.add_argument('--use-bn', action='store_true', help='Use batch normalization in intermediate layers')
	parser.add_argument('-o', '--optimizer', default='adam', type=str,
	                    metavar='OPTIM', help='type of optimization: sgd | adam | [rmsprop]')
	parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
	                    metavar='LR', help='initial learning rate (default: 1e-4)')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
	                    help='momentum (default: 0.9)')
	parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='M',
	                    help='weight decay (default: 0.9)')

	parser.add_argument('--buffer-size', type=int, default=100000,
	                    help='Size of the replay buffer from which transitions are sampled to train on')
	parser.add_argument('--buffer-start', type=int, default=500,
	                    help='Size of the replay buffer at which training begins')
	parser.add_argument('--discount', type=float, default=0.99, help='RL discount factor for future rewards')
	parser.add_argument('--tau', type=float, default=1e-3, help='EMA parameter for DDPG target net')
	parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')

	parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Don\'t use cuda')
	parser.add_argument('--seed', type=int, default=5, help='Seed for random state')
	parser.add_argument('--episodes', type=int, required=True, help='Number of episodes to run now (required)')
	parser.add_argument('--save-freq', type=int, default=20, help='Number of episodes in between saving stats and net')
	parser.add_argument('--test-len', type=int, default=10, help='Number of episodes to test (default 10)')
	parser.add_argument('--test-freq', type=int, default=20,
	                    help='Number of episodes of training in between testing (0 means no testing) (default 20)')
	parser.add_argument('--print-freq', type=int, default=1, help='Number of episodes between printing')

	# parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
	# parser.add_argument('--gae-lambda', type=float, default=0.97, help='Model Parameter')
	# parser.add_argument('--hidden-sizes', type=int, default=(32,32), nargs='+', help='Sizes of hidden layers for policy')
	# parser.add_argument('--init-log-std', type=float, default=-0.1, help='Initial log std for policy')
	# parser.add_argument('--step-size', type=float, default=0.05, help='Step size for agent')

	# parser.add_argument('--num-agents', type=int, default=4, help='Number of agents')
	# parser.add_argument('--num-particles', type=int, default=1, help='Number of particles')
	# parser.add_argument('--grid-side', type=int, default=11, help='Size of the grid')
	# parser.add_argument('--obs-grid', action='store_true', help='Observe a grid instead of coords')
	# parser.add_argument('--expressive-rewards', action='store_true', help='Use more expressive rewards')

	# parser.add_argument('--')

	# parser.add_argument('--control', type=str, default='vel', choices=['force', 'vel'], help='What type of control to use')
	# parser.add_argument('--goal', type=str, default='trap', choices=['target', 'trap', 'trajectory'], help='What goal to use')
	# parser.add_argument('--noise', type=float, default=0, help='mag of noise')
	# parser.add_argument('--planning', action='store_true', help='Use MCTS for planning')
	# parser.add_argument('--curiosity', action='store_true', help='Use curiosity to improve exploration')

	return parser



