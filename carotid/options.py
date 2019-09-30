


import configargparse


def setup_classify_options():
	
	parser = configargparse.ArgumentParser(description='Formatting and Preprocessing: mp3 -> hdf')
	
	parser.add_argument('-c', '--config', required=True, is_config_file=True,
						help='Path to config file for parameters')
	
	parser.add_argument('--dataset', type=str, help='Path to folder containing tar files with pose data')
	parser.add_argument('--data-mode', type=str, default='genre')
	parser.add_argument('-d', '--data', nargs='+',
						help='Path to folder containing tar files with pose data')
	parser.add_argument('-s', '--save-root', default='trained_nets', type=str, metavar='PATH',
						help='directory to save results in. If it doesnt exist, will be created. (default: results/)')
	parser.add_argument('-n', '--name', type=str, default='test')
	
	parser.add_argument('-b', '--batch-size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--train-per', default=0.6, type=float,
						metavar='FRAC', help='fraction of data for the training set (default: 0.6)')
	parser.add_argument('--val-per', default=0.15, type=float,
						metavar='FRAC', help='fraction of data for the validation set (default: 0.15)')
	parser.add_argument('--split-path', type=str, default='test')
	
	# Data options
	parser.add_argument('--sample-rate', type=int, default=-1)
	parser.add_argument('--step-len', default=1, type=int, metavar='N',
						help='number of frames separating each example in the training sequence (default: 1)')
	parser.add_argument('--seq-len', default=1, type=int,
						metavar='N', help='length of the training sequence (default: 1)')
	parser.add_argument('--hop', default=10, type=int,)
	parser.add_argument('--window', default=50, type=int, )
	
	parser.add_argument('-j', '--num-workers', default=0, type=int, metavar='N',
						help='dimensionality of the control space (default: 7)')
	
	parser.add_argument('--data-type', type=str) # spec vs wav
	parser.add_argument('--model', type=str, default='naive', help='type of transition net model to train')
	parser.add_argument('--hidden-dims', type=int, nargs='+')
	parser.add_argument('--no-cuda', dest='cuda', action='store_false')
	
	parser.add_argument('--model-hop', type=int, default=10)
	parser.add_argument('--model-ws', type=int, default=50)
	parser.add_argument('--model-n', type=int, default=256)
	
	parser.add_argument('--mel-hop', type=int, default=10)
	parser.add_argument('--mel-ws', type=int, default=50)
	parser.add_argument('--mel-n', type=int, default=128)
	
	parser.add_argument('--gen-hop', type=int, default=10)
	parser.add_argument('--gen-ws', type=int, default=50)
	parser.add_argument('--gen-n', type=int, default=128)
	
	parser.add_argument('--latent-dim', type=int, default=0)
	parser.add_argument('--rec-loss', type=str, default='mse')
	parser.add_argument('--rec-loss-wt', type=float, default=1e-2)
	parser.add_argument('--cls-loss-wt', type=float, default=1)
	parser.add_argument('--var-loss-wt', type=float, default=1e-2) # beta
	parser.add_argument('--noisy-rec', action='store_true')
	
	parser.add_argument('--nonlin', type=str, default='prelu')
	parser.add_argument('--rec-dim', type=int, default=128)
	parser.add_argument('--rec-num-layers', type=int, default=1)
	parser.add_argument('--rec-type', type=str, default='gru')
	parser.add_argument('--fc-dims', type=int, nargs='+', default=None)
	parser.add_argument('--use-fc', action='store_true')
	parser.add_argument('--use-batch-norm', action='store_true')
	parser.add_argument('--dense-labeling', action='store_true')
	parser.add_argument('--full-records', action='store_true')
	parser.add_argument('--single-step', action='store_true')
	
	parser.add_argument('--gen-rec-dim', type=int, default=128)
	parser.add_argument('--gen-rec-num-layers', type=int, default=3)
	parser.add_argument('--gen-use-cls', action='store_true')
	parser.add_argument('--gen-steps', type=int, default=1)
	parser.add_argument('--gen-gain', type=float, default=10)
	parser.add_argument('--gen-norm', action='store_true')
	#parser.add_argument('--judge-mel', action='store_true')
	parser.add_argument('--judge-steps', type=int, default=1)
	parser.add_argument('--judge-clip', type=float, default=0)
	
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--epochs', default=100, type=int, metavar='N',
						help='number of total epochs to run (default: 100)')
	parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
						help='manual epoch number (useful on restarts) (default: 0)')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
						help='evaluate model on test set (default: False)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	
	# Optimization options
	parser.add_argument('--loss-type', type=str, default='mse')
	parser.add_argument('--loss-scale', default=1, type=float,)
	parser.add_argument('-o', '--optimization', default='rmsprop', type=str,
						metavar='OPTIM', help='type of optimization: sgd | adam | [rmsprop]')
	parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
						metavar='LR', help='initial learning rate (default: 1e-4)')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='momentum (default: 0.9)')
	parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)')
	parser.add_argument('--lr-decay', default=0.1, type=float, metavar='M',
						help='Decay learning rate by this value every decay-epochs (default: 0.1)')
	parser.add_argument('--decay-epochs', default=30, type=int,
						metavar='M', help='Decay learning rate every this many epochs (default: 10)')
	
	# Display/Save options
	parser.add_argument('--disp-freq', '-p', default=25, type=int,
						metavar='N', help='print/disp/save frequency (default: 25)')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	parser.add_argument('--tblog', action='store_true')
	parser.add_argument('--txtlog', action='store_true')
	parser.add_argument('--log-date', action='store_true')
	parser.add_argument('--no-test', action='store_true')
	
	
	
	
	return parser

def setup_format_options():

	parser = configargparse.ArgumentParser(description='Formatting and Preprocessing: mp3 -> hdf')

	parser.add_argument('-c', '--config', required=True, is_config_file=True,
	                    help='Path to config file for parameters')

	parser.add_argument('-i', '--input-dir', type=str, default=None)
	parser.add_argument('-r', '--recursive', action='store_true', )
	parser.add_argument('--input-files', type=str, nargs='+', default=None)
	parser.add_argument('--input-path-file', type=str, default=None)

	parser.add_argument('--output-dir', type=str, required=True, default=None)
	parser.add_argument('--output-template', type=str, default='track{}.h5')
	parser.add_argument('--meta-path', type=str, default=None)

	parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
	                    help='number of data loading workers (default: 4)')

	parser.add_argument('--normalize', action='store_true', )
	parser.add_argument('--cut-zeros', action='store_true', )

	return parser


