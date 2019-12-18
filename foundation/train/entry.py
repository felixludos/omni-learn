

import sys, os, time
# import ipdb
import traceback

from contextlib import nullcontext, redirect_stdout, redirect_stderr

from .config import parse_config
from .model import default_create_model
from .data import default_load_data

from .running import run_full

def main(config=None, argv=None, get_model=None, get_data=None, get_name=None):
	# WARNING: 'argv' should be equivalent to sys.argv here (with script name in element 0)

	ctxt, ctxt2 = nullcontext(), nullcontext()

	if 'FOUNDATION_RUN_MODE' not in os.environ:
		mode = 'cmd'
		os.environ['FOUNDATION_RUN_MODE'] = mode
		print('WARNING: $FOUNDATION_RUN_MODE is not set, using default: {}'.format(mode))
	mode = os.environ['FOUNDATION_RUN_MODE']

	if 'FOUNDATION_TESTING' not in os.environ:
		testing = '1'
		os.environ['FOUNDATION_TESTING'] = testing
		print('WARNING: $FOUNDATION_TESTING is not set, using default {}'.format(testing))
	testing = os.environ['FOUNDATION_TESTING'] == '1'


	assert mode in {'cmd', 'jupyter', 'cluster', 'pycharm'}, 'Unknown run mode: {}'.format(mode)

	if mode == 'cluster' and 'JOBDIR' in os.environ:
		ctxt = redirect_stderr(sys.stdout)
		ctxt2 = redirect_stdout(open(os.path.join(os.environ['JOBDIR'], 'out{}.log'.format(os.environ['PROCESS_ID'])), 'a+'))

	with ctxt, ctxt2:

		if config is None:

			if testing and mode == 'cmd':
				print('-- Inserting \'cmd\' to front of config (because of testing mode) --')
				argv.insert(1,'cmd')

			config = parse_config(argv[1:])

		if os.environ['FOUNDATION_TESTING'] == '1':
			print('\nThis is a test run!\n')

		if 'name' not in config and mode in {'cmd', 'pycharm'} and 'test_override' not in config:
			config.name = 'test-{}'.format(mode)
			print('Name defaulting to: {}'.format(config.name))


		config.run_mode = mode

		assert 'FOUNDATION_SAVE_DIR' in os.environ, 'no save dir provided'
		config.saveroot = os.environ['FOUNDATION_SAVE_DIR']
		print('Set saveroot to: {}'.format(config.saveroot))

		assert 'FOUNDATION_DATA_DIR' in os.environ, 'no data dir provided'
		config.dataroot = os.environ['FOUNDATION_DATA_DIR']
		print('Set dataroot to: {}'.format(config.dataroot))

		if get_model is None:
			get_model = default_create_model
		if get_data is None:
			get_data = default_load_data

		if mode == 'cluster' and 'JOBDIR' in os.environ: # TODO: setup links for restarts

			cname = 'checkpoints{}.txt'.format(os.environ['PROCESS_ID'])

			if 'auto_name' in config:

				ID = os.environ['JOB_ID'].split('#')[-1].split('.')[0]
				ps = os.environ['PROCESS_ID']
				num = os.environ['JOB_NUM']

				terms = []

				if 'dataset_type' in config.info:
					terms.append(config.info.dataset_type)
				elif 'name' in config.dataset:
					terms.append(config.dataset.name)

				if 'model_type' in config.info:
					terms.append(config.info.model_type)
				elif '_type' in config.model:
					terms.append(config.model._type)


				autoname = '-'.join(terms) if len(terms) else 'run'

				prefix = config.name if 'name' in config else autoname

				config.name = '{}_{}-{}-{}'.format(prefix, str(num).zfill(4), ID, str(ps).zfill(2))
				config.info.job.ID = ID
				config.info.job.num = num
				config.info.job.ps = ps


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
				import ipdb
				extype, value, tb = sys.exc_info()
				traceback.print_exc()
				ipdb.post_mortem(tb)
			else:
				raise e

	return 0
