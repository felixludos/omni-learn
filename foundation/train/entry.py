

import sys, os, time
import ipdb
import traceback

from contextlib import nullcontext, redirect_stdout, redirect_stderr

from .config import parse_cmd_args
from .model import default_create_model
from .data import default_load_data

from .running import run_full


def main(config=None, argv=None, get_model=None, get_data=None, get_name=None):

	ctxt, ctxt2 = nullcontext(), nullcontext()

	if 'FOUNDATION_RUN_MODE' not in os.environ:
		mode = 'cmd'
		os.environ['FOUNDATION_RUN_MODE'] = mode
		print('WARNING: $FOUNDATION_RUN_MODE is not set, using default: {}'.format(mode))

	else:
		mode = os.environ['FOUNDATION_RUN_MODE']

	assert mode in {'cmd', 'jupyter', 'cluster', 'pycharm'}, 'Unknown run mode: {}'.format(mode)

	if mode == 'cluster' and 'JOBDIR' in os.environ:
		ctxt = redirect_stderr(sys.stdout)
		ctxt2 = redirect_stdout(open(os.path.join(os.environ['JOBDIR'], 'out{}.log'.format(os.environ['PROCESS_ID'])), 'a+'))

	with ctxt, ctxt2:

		if config is None:
			config = parse_cmd_args(argv)

		config.run_mode = mode

		if 'saveroot' not in config and 'FOUNDATION_SAVE_DIR' in os.environ:
			config.saveroot = os.environ['FOUNDATION_SAVE_DIR']
			print('Set saveroot to: {}'.format(config.saveroot))

		if 'dataroot' not in config and 'FOUNDATION_DATA_DIR' in os.environ:
			config.dataroot = os.environ['FOUNDATION_DATA_DIR']
			print('Set dataroot to: {}'.format(config.dataroot))

		if get_model is None:
			get_model = default_create_model
		if get_data is None:
			get_data = default_load_data

		if mode == 'cluster' and 'JOBDIR' in os.environ: # TODO: setup links for restarts

			cname = 'checkpoints{}.txt'.format(os.environ['PROCESS_ID'])

			if cname in os.listdir(os.environ['JOBDIR']):
				print('This job has already made progress')
				with open(os.path.join(os.environ['JOBDIR'], cname), 'r') as f:
					config.resume = f.readline()

				print('Resuming: {}'.format(config.resume))


		try:
			run_full(config, get_data, get_model, get_name=get_name)

		except KeyboardInterrupt:
			extype, value, tb = sys.exc_info()
			traceback.print_exc()

		except Exception as e:
			if mode == 'cmd':
				extype, value, tb = sys.exc_info()
				traceback.print_exc()
				ipdb.post_mortem(tb)
			else:
				raise e

	return 0
