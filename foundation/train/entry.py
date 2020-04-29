

import sys, os, time
import inspect
# import ipdb
import traceback
import socket

from contextlib import nullcontext, redirect_stdout, redirect_stderr

from .registry import view_script_registry, autofill_args
from .config import parse_config


from .running import iterative_run

def main(config=None, argv=None, cmd=None, **cmd_kwargs):
	'''
	Should be the entry point of ALL scripts that use foundation (especially those that use Configs).

	:param config: root config
	:param argv: equivalent to sys.argv (with script name in element 0)
	:param cmd: callable takes as input the loaded config and any cmd_kwargs that are provided
	:param cmd_kwargs: any kwargs that (generally this should only include fixed functions to process the config in cmd)
	:return: the output of running cmd with the loaded config and cmd_kwargs
	'''

	if cmd is None:
		cmd = iterative_run

	ctxt, ctxt2 = nullcontext(), nullcontext()

	if 'FOUNDATION_RUN_MODE' not in os.environ:
		mode = 'cmd'
		os.environ['FOUNDATION_RUN_MODE'] = mode
		print('WARNING: $FOUNDATION_RUN_MODE is not set, using default: {}'.format(mode))
	mode = os.environ['FOUNDATION_RUN_MODE']

	# mode = 'cluster'
	# print('\n\n\n\n\n\nTESTING: running on cluster')

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

			config = parse_config(argv[1:], include_load_history=True)

			assert '_history' in config, 'should not happen'
			if '_history' in config:
				config.info.history = config._history
				del config._history

			config.info.argv = argv#[1:]

		if 'test_mode' in config:
			os.environ['FOUNDATION_TESTING'] = str(int(config.test_mode))
			print('Test mode set to: {}'.format(config.test_mode))

		if os.environ['FOUNDATION_TESTING'] == '1':
			print('\nThis is a test run!\n')


		if '_loaded' in config:
			print('Loaded config from: {}'.format(config._loaded))

			if 'name' in config:
				del config.name

			if 'output' in config:
				if '_logged_date' in config.output:
					del config.output._logged_date
				if 'name' in config.output:
					del config.output.name
			if 'save_dir' in config:
				del config.save_dir
			# del config.info

			# if 'name' in config:
			# 	del config.name
			# if 'output' in config and '_logged_date' in config.output:
			# 	del config.output._logged_date
			# if 'info' in config:
			# 	del config.info

		if 'name' not in config and os.environ['FOUNDATION_TESTING'] == '1':
			config.name = 'test-{}'.format(mode)
			print('Name defaulting to: {}'.format(config.name))


		config.run_mode = mode

		assert 'FOUNDATION_SAVE_DIR' in os.environ, 'no save dir provided'
		config.saveroot = os.environ['FOUNDATION_SAVE_DIR']
		print('Set saveroot to: {}'.format(config.saveroot))

		assert 'FOUNDATION_DATA_DIR' in os.environ, 'no data dir provided'
		config.dataroot = os.environ['FOUNDATION_DATA_DIR']
		print('Set dataroot to: {}'.format(config.dataroot))



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
				
				if 'extra' in config.info:
					autoname = '{}-{}'.format(autoname, config.info.extra)

				prefix = config.name if 'name' in config else autoname

				config.name = '{}_{}-{}-{}'.format(prefix, str(num).zfill(4), ID, str(ps).zfill(2))
				config.info.job.ID = ID
				config.info.job.num = num
				config.info.job.ps = ps

			print('Hostname: {}{}'.format(socket.gethostname(),
			                                ', ' + os.environ['CUDA_VISIBLE_DEVICES']
			                                if 'CUDA_VISIBLE_DEVICES' in os.environ
			                                else ''))
			
			# Check if this job has already made some progress, and if so, resume.
			if cname in os.listdir(os.environ['JOBDIR']):
				print('This job has already made progress')
				with open(os.path.join(os.environ['JOBDIR'], cname), 'r') as f:
					config.resume = f.readline()

				print('Resuming: {}'.format(config.resume))

		out = 0
		try:
			out = cmd(config, **cmd_kwargs)

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

	return out


_help_msg = '''fdrun <script> [args...]
Please specify a script (and optionally args), registered scripts:
{}'''
_help_cmds = {'-h', '--help'}

_error_msg = '''Error script {} is not registered.
Please specify a script (and optionally args), registered scripts:
{}'''


def main_script(argv=None):
	if argv is None:
		argv = sys.argv[1:]
	
	scripts = view_script_registry()
	script_names = ', '.join(scripts.keys())
	
	if len(argv) == 0 or (len(argv) == 1 and argv[0] in _help_cmds):
		print(_help_msg.format(script_names))
		return 0
	elif argv[0] not in scripts:
		print(_error_msg.format(argv[0], script_names))
		return 1
	
	name, *argv = argv
	fn, use_config = scripts[name]
	
	if len(argv) == 1 and argv[0] in _help_cmds:
		print(f'Help message for script: {name}')
		
		doc = fn.__doc__
		
		if doc is None and not use_config:
			doc = str(inspect.signature(fn))
			doc = f'Arguments {doc}'
		
		print(doc)
		return 0
	
	A = parse_config(argv=argv)
	
	if use_config:
		out = fn(A)
	else:
		out = autofill_args(fn, A)

	if out is None:
		return 0
	return out

