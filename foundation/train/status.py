
import sys, os, shutil
import yaml
import subprocess
from datetime import datetime
from tabulate import tabulate
import numpy as np

from .. import util

from .registry import Script

colattrs = ['ClusterId', 'ProcId', 'JobStatus', 'Args', 'RemoteHost']

_status_codes = {
	'0': 'Unexpanded',
	'1': 'Idle',
	'2': 'Running',
	'3': 'Removed',
	'4': 'Completed',
	'5': 'Held',
	'6': 'Submission_err',

}

def parse_jobexec(raw, info=None):
	*root, jdir, jexe = raw.split('/')
	
	num, date = jdir.split('_')
	num = int(num[3:])
	
	if info is None:
		info = util.tdict()
	
	info.num = num
	info.raw_date = date
	info.date = datetime.strptime(date, '%y%m%d-%H%M%S')
	info.str_date = info.date.ctime()#.strftime()
	
	info.exe = jexe
	info.path = os.path.dirname(raw)
	
	return info

def parse_remotehost(raw):
	s, g = raw.split('.')[0].split('@')
	s = s.split('_')[-1]
	return f'{s}{g}'

def parse_job_status(info):
	if 'ClusterId' in info:
		info.job_id = info.ClusterId
		del info.ClusterId
		
	if 'ProcId' in info:
		info.proc_id = int(info.ProcId)
		del info.ProcId
		
	if 'JobStatus' in info:
		info.status = _status_codes[info.JobStatus]
		del info.JobStatus
		
	if 'Args' in info:
		parse_jobexec(info.Args, info)
		del info.Args
	
	if 'RemoteHost' in info:
		try:
			info.host = parse_remotehost(info.RemoteHost)
		except Exception:
			info.host = info.RemoteHost
		del info.RemoteHost
	
	return info

def collect_q_cmd():
	print('Getting job status...', end='')
	
	# raw = subprocess.check_output(['condor_q', 'fleeb', '-af', 'ClusterId', 'ProcId', 'Args', 'JobStatus', 'RemoteHost', 'Env'])
	raw = subprocess.check_output(['condor_q', 'fleeb', '-af:t'] + colattrs).decode()
	
	if len(raw) == 0:
		print(' No jobs running.')
		return None
	else:
		lines = raw.split('\n')
		print(' found {} jobs'.format(len(lines)))
	
	# print(lines)
	
	R = util.MultiDict(parse_job_status(util.tdict(zip(colattrs, line.split('\t')))) for line in lines if len(line))
	
	return R

def peek_file(opath, peek=0):
	if os.path.isfile(opath):
		with open(opath, 'r') as f:
			return f.readlines()[-peek:]
	return None

def print_current(full, simple=True):
	table = []
	for info in full:
		table.append([f'{info.job_id}.{info.proc_id}', f'{info.host}', f'{info.num}-{info.proc_id}',
		              f'{info.str_date}', f'{info.status}'])
		
	if simple:
		print(tabulate(table, ['ClusterId', 'Host', 'JobId', 'StartDate', 'Status'], floatfmt='.10g'))
	
	else:
		for row, info in zip(table, full):
			head = '----- {} -----'.format(' -- '.join(row))
			tail = '-'*len(head)
			
			print(head)
			print(''.join(info.peek),end='')
			print(tail)
			print()

			
@Script('cls')
def get_status(peek=None):
	'''
	Script to get a status of the cluster jobs
	'''
	
	# assert 'FOUNDATION_RUN_MODE' in os.environ and os.environ['FOUNDATION_RUN_MODE'] == 'cluster', 'Should be the cluster'
	
	print()
	
	current = collect_q_cmd()
	
	print()
	
	if current is None or len(current) == 0:
		# print('No jobs running.')
		return 0
	
	if peek is not None:
		for info in current:
			if 'path' in info and 'num' in info:
				opath = os.path.join(info.path, 'out{}.log'.format(info.proc_id))
				# print(opath)
				info.peek = peek_file(opath, peek)
	
	print_current(current, simple=peek is None)
	
	return 0



def parse_run_name(raw, info=None):
	if info is None:
		info = util.tdict()
	
	name, job, date = raw.split('_')

	info.jname = name
	
	info.job, info.cls, info.num = map(int, job.split('-'))
	
	info.date = datetime.strptime(date, '%y%m%d-%H%M%S')
	info.str_date = info.date.ctime()  # .strftime()

	return info

def collect_runs(path, recursive=False, since=None, last=5):
	
	if recursive:
		raise NotImplementedError
	
	runs = util.MultiDict()
	
	for rname in os.listdir(path):
		rpath = os.path.join(path, rname)
		if os.path.isdir(rpath):
			if recursive:
				child = collect_runs(rpath, recursive=recursive)
				runs.extend(child)
			else:
				info = util.tdict(name=rname, path=rpath)
				
				try:
					parse_run_name(rname, info)
				except Exception as e:
					raise e
				
				runs.append(info)
	
	if since is None:
		assert last is not None
		
		sel = list(sorted(set(runs.by('job'))))
		if len(sel) > last:
			sel = sel[-last:]
		
	else:
		sel = {x for x in runs.by('job') if x >= since}
	
	print('Including jobs:', sel)
	
	# print(runs[0])
	# print(runs[0].keys())
	# print(runs[0].values())
	
	runs.filter(lambda run: run.job in sel)
	
	return runs

def load_runs(runs, load_configs=False):
	
	# print(runs)
	
	for run in runs:
		# print(run)
		# print(dict(run))
		contents = os.listdir(run.path)
		if 'config.yml' in contents:
			
			ckpts = [int(name.split('.')[0].split('_')[-1]) for name in contents if 'checkpoint' in name]
			run.done = max(ckpts) if len(ckpts) > 0 else 0
			run.num_ckpts = len(ckpts)
			
			if load_configs:
				run.config = yaml.load(open(os.path.join(run.path, 'config.yml')))
		
		else:
			run.no_config = True
			
			if load_configs:
				run.config = None

# _run_status = ['Completed', 'Error', 'Failed', ]

def evaluate_status(runs, active=None, cmpl=None):
	for run in runs:
		try:
			
			if 'no_config' in run and run.no_config:
				run.status = 'NoConfig'
			
			elif active is not None and run.name in active:
				run.status = active[run.name].status
			
			else:
				run.status = 'Failed'

			target = None
			if 'config' in run and 'training' in run.config and 'step_limit' in run.config['training']:
				target = run.config['training']['step_limit']
			elif cmpl is not None:
				target = cmpl
			
			run.progress = -1
			if target is not None:
				if 'done' not in run:
					run.status = 'Error'
				else:
					run.progress = run.done / target
					if run.progress >= 1:
						run.status = 'Completed'
					
		except Exception as e:
			run.status = 'Error'
			raise e
		
def print_table(table, cols=None, name=None):
	print()
	print('-' * 50)
	if name is not None:
		print(name)
	
	if len(table) == 0:
		print('None')
	else:
		print(tabulate(table, cols))
	
	print('-' * 50)
	
def print_run_status(runs):
	
	success = []
	running = []
	fail = []
	
	for run in runs:
		if 'status' not in run:
			fail.append(run)
		elif run.status == 'Completed':
			success.append(run)
		elif run.status in _status_codes.values():
			running.append(run)
		else:
			fail.append(run)
	
	assert len(success) + len(running) + len(fail) == len(runs), f'{len(success)}, {len(running)}, ' \
	                                                             f'{len(fail)} vs {len(runs)}'
	
	# sort runs
	
	success = sorted(success, key=lambda r: r.date)
	running = sorted(running, key=lambda r: r.progress)
	fail = sorted(fail, key=lambda r: r.progress)
	
	# completed jobs
	
	cols = ['Name', 'Date', ]
	
	rows = []
	for run in success:
		row = [run.name, run.date, ]
		rows.append(row)
	
	if 'config' in run:
		cols.append('Command')
		for row, run in zip(rows, runs):
			row.append(' '.join(run.config['info']['argv']))
	
	print_table(rows, cols, 'Completed jobs:')
	
	# running jobs
	
	cols = ['Name', 'Date', 'Progress', 'Status']
	
	rows = []
	for run in running:
		row = [run.name, run.date, f'{run.progress * 100:3.1f}', run.status]
		rows.append(row)
	
	if 'config' in run:
		cols.append('Command')
		for row, run in zip(rows, runs):
			row.append(' '.join(run.config['info']['argv']))
	
	print_table(rows, cols, 'Running jobs:')
	
	# failed jobs
	
	cols = ['Name', 'Date', 'Progress', 'Status']
	
	rows = []
	for run in fail:
		row = [run.name, run.date, f'{run.progress * 100:3.1f}', run.status]
		rows.append(row)
	
	if 'config' in run:
		cols.append('Command')
		for row, run in zip(rows, runs):
			row.append(' '.join(run.config['info']['argv']))
	
	print_table(rows, cols, 'Failed jobs:')
	
	
	# summary
	
	print()
	print(tabulate([
		['Completed', len(success)],
		['Running', len(running)],
		['Failed', len(fail)],
	]))
	

@Script('status')
def check_runs(since=None, last=5, running=True,
               recursive_runs=False,
               load_configs=False, def_cmpl=100000, path=None):
	'''
	Check the status of the saved runs
	'''
	
	print()
	
	if path is None:
		assert 'FOUNDATION_SAVE_DIR' in os.environ
		path = os.environ['FOUNDATION_SAVE_DIR']
	
	runs = collect_runs(path, recursive=recursive_runs, since=since, last=last)
	
	if len(runs) == 0:
		print('No runs found.')
		return 0
	else:
		print(f'Found {len(runs)} runs')
	
	load_runs(runs, load_configs=load_configs)
	
	
	active = None
	if running:
		print()
		current = collect_q_cmd()
		
		active = util.tdict()
		
		for info in current:
			cname = f'checkpoints{info.proc_id}.txt'
			if cname in os.listdir(info.path):
				with open(os.path.join(info.path, cname), 'r') as f:
					info.run_name = f.read()
				active[info.run_name] = info
		print()
	
	evaluate_status(runs, active, cmpl=def_cmpl)
	
	print_run_status(runs)
	
	return 0


