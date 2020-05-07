
import sys, os, shutil
import yaml
import subprocess
from datetime import datetime
from tabulate import tabulate
import numpy as np

from .. import util
from .config import get_config

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
	
	info.jnum = num
	# info.raw_date = date
	info.date = datetime.strptime(date, '%y%m%d-%H%M%S')
	# info.str_date = info.date.ctime()#.strftime()
	
	info.jexe = jexe
	info.path = os.path.dirname(raw)
	
	return info

def parse_remotehost(raw):
	s, g = raw.split('.')[0].split('@')
	s = s.split('_')[-1]
	return f'{s}{g}'

def parse_job_status(raw):
	
	info = util.tdict()
	
	if 'ClusterId' in raw:
		info.ID = int(raw.ClusterId)
		
	if 'ProcId' in raw:
		info.proc = int(raw.ProcId)
		
	if 'JobStatus' in raw:
		info.status = _status_codes[raw.JobStatus]
		
	if 'Args' in raw:
		parse_jobexec(raw.Args, info)
	
	if 'RemoteHost' in raw:
		try:
			info.host = parse_remotehost(raw.RemoteHost)
		except Exception:
			info.host = raw.RemoteHost
	
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
		print(' found {} jobs'.format(len(lines)-1))
	
	# print(lines)
	
	active = util.Table(parse_job_status(util.tdict(zip(colattrs, line.split('\t')))) for line in lines if len(line))
	
	return active

def peek_file(opath, peek=0):
	if os.path.isfile(opath):
		with open(opath, 'r') as f:
			return f.readlines()[-peek:]
	return None

def print_current(full, simple=True):
	table = []
	for info in full:
		table.append([f'{info.job_id}.{info.proc_id}', f'{info.host}', f'{info.ID}-{info.proc_id}',
		              f'{info.str_date}', f'{info.status}'])
		
	if simple:
		print(tabulate(table, ['ClusterId', 'Host', 'JobId', 'StartDate', 'Status'], floatfmt='.10g', disable_numparse=True))
	
	else:
		for row, info in zip(table, full):
			head = '----- {} -----'.format(' -- '.join(row))
			tail = '-'*len(head)
			
			print(head)
			if info.peek is not None:
				print(''.join(info.peek),end='')
			else:
				print('[no output file found]')
			print(tail)
			print()

def load_registry(path, last=5, since=None):
	with open(os.path.join(path, 'registry.txt'), 'r') as f:
		lines = f.readlines()
		
	keys = ['JobID', 'HostName', 'RunName', 'JobDir']
	terms = [[w.strip(' ').strip('\n') for w in row.split(' - ')] for row in lines]

	jobs = util.Table()
	for row in terms:
		raw = util.tdict(zip(keys, row))
		info = jobs.new()

		info.ID, info.proc = map(int, raw.JobID.split('.'))
		
		info.name = raw.JobDir
		info.host = raw.HostName
		info.rname = raw.RunName
		
		words = raw.JobDir.split('_')
		info.jname = '_'.join(words[:-1])
		info.date = datetime.strptime(words[-1], '%y%m%d-%H%M%S')

		info.path = os.path.join(path, info.name)

		try:
			if info.jname.startswith('job'):
				info.jnum = int(info.jname.strip('job'))
		except Exception as e:
			raise e
			info.jnum = None

	available = set(jobs.select('ID'))
	
	if since is not None:
		
		lim = None
		if since in available:
			lim = since
		else:
			for info in jobs:
				if since in {info.jname, info.name, info.jnum}:
					lim = info.ID
		assert lim is not None, f'{since} not found'
		
		jobs.filter_(lambda info: info.ID >= lim)
		
	else:
		options = sorted(map(int, available))
		accepted = set(options[-last:])
		jobs.filter_(lambda x: x.ID in accepted)
	
	mpath = os.path.join(path, 'manifest.txt')
	if os.path.isfile(mpath):
		with open(mpath, 'r') as f:
			lines = [line.split(' - ') for line in f.read().split('\n')]
		
		counts = {}
		for row in lines:
			if len(row) == 3:
				counts[row[0]] = int(row[1])
			# else:
			# 	print(f'Failed: {row}')
		
		nums = {}
		present = {}
		
		for info in jobs:
			
			if info.name not in present:
				present[info.name] = [None]*counts[info.name]
			
			if info.name not in nums:
				nums[info.name] = info.ID
			
			if present[info.name][info.proc] is not None:
				raise Exception(f'Duplicate: {info.name} {info.ID}.{info.proc}')
		
			present[info.name][info.proc] = info
			
		for name, procs in present.items():
			for proc, info in enumerate(procs):
				if info is None:
					print(f'Missing: {name} {nums[name]} {proc}')
					jobs.new(name=name, ID=nums[name], proc=proc, status='Missing',
					         path=os.path.join(path, name), )
	
	return jobs

def connect_current(jobs, current):
	# print(list(map(dict,jobs)))
	IDs = {(j.ID, j.proc): j for j in jobs}
	
	for run in current:
		if (run.ID, run.proc) in IDs:
			job = IDs[run.ID, run.proc]
			job.host = run.host
			job.status = run.status
		else:
			print(run.ID, run.proc, 'not found')
			jobs.append(run)
			run.section = 'extra'
	
	# print(4, len(jobs))
	
	return jobs

def connect_saves(jobs, saveroot=None, load_configs=False):
	
	if saveroot is None:
		saveroot = os.environ['FOUNDATION_SAVE_DIR']
	
	# rnames = set(os.listdir(saveroot))
	
	for info in jobs:
		cname = os.path.join(info.path, f'checkpoints{info.proc}.txt')
		if 'rname' not in info and os.path.isfile(cname):
			with open(cname, 'r') as f:
				info.rname = f.read()
	
		if 'rname' in info:
			
			info.rpath = os.path.join(saveroot, info.rname)
			
			if os.path.isdir(info.rpath):
				if load_configs:
					with open(os.path.join(saveroot, info.rname, 'config.yml')) as f:
						info.config = yaml.load(f)
				
				contents = os.listdir(info.rpath)
				ckpts = [int(name.split('.')[0].split('_')[-1]) for name in contents if 'checkpoint' in name]
				info.done = max(ckpts) if len(ckpts) > 0 else 0
				info.num_ckpts = len(ckpts)
			
			else:
				info.done = 0
				info.status = 'Error'
				info.error_msg = f'Run path not found: {info.rpath}'
			
	return jobs

def fill_status(jobs, target=None):
	for info in jobs:
		if 'config' in info:
			info.target = info.config['training']['step_limit']
		elif target is not None:
			info.target = target
		
		if 'target' in info and 'done' in info:
			info.progress = info.done / info.target
		
		if 'status' not in info:
			info.status = 'Completed' if 'progress' in info and info.progress >= 1 else 'Failed'
			
	return jobs


def print_table(table, cols=None, name=None):
	print()
	# print('-' * 50)
	if name is not None:
		print(name)
	
	if len(table) == 0:
		print('None')
	else:
		print(tabulate(table, cols, disable_numparse=True))

def print_separate_table(table, cols=None, name=None, peeks=None):
	print()
	
	if name is not None:
		print(name)
	
	if peeks is None:
		peeks = [None]*len(table)
	
	if len(table) == 0:
		print('None')
	else:
		for row, peek in zip(table, peeks):
			msg = '--- {} ---'.format(' - '.join(map(str,row)))
			print(msg)
			if peek is not None:
				print(''.join(peek),end='')
			else:
				print('[no output file found]')
			print('-'*len(msg))
			print()

def print_status(jobs, list_failed=False, show_peeks=None, skip_missing=None):
	success = []
	running = []
	fail = []
	errors = []
	
	# print(len(jobs))
	
	for info in jobs:
		if 'status' not in info:
			fail.append(info)
		elif info.status == 'Completed':
			success.append(info)
		elif info.status in _status_codes.values():
			running.append(info)
		elif skip_missing and info.status == 'Error':
			errors.append(info)
		else:
			fail.append(info)
	
	assert len(success) + len(running) + len(fail) + len(errors) == len(jobs), f'{len(success)}, {len(running)}, ' \
	                                                             f'{len(fail)}, {len(errors)} vs {len(jobs)}'
	
	# sort runs
	
	success = sorted(success, key=lambda r: r.date)
	# running = sorted(running, key=lambda r: r.progress if 'progress' in r else 0)
	running = sorted(running, key=lambda r: (r.ID, r.proc))
	fail = sorted(fail, key=lambda r: r.progress if 'progress' in r else 0)
	
	if skip_missing:
		print('Errors')
		print('\n'.join(e.rname for e in errors))
	
	# print(len(success),len(running),len(fail))
	
	cols = ['Name', 'Date', 'Progress']
	rows = []
	peeks = []
	for info in success:
		row = [info.rname if 'rname' in info else info.name, info.date,
		       f'{info.done//1000}/{info.target//1000 if isinstance(info.target, int) else info.target}',]
		rows.append(row)
		peeks.append(info.peek if 'peek' in info else None)
	title = 'Completed jobs:'
	# if show_peeks:
	# 	print_separate_table(rows, cols, title, peeks)
	# else:
	# 	print_table(rows, cols, title)
	print_table(rows, cols, title)
	
	cols = ['Name', 'Date', 'Progress', 'Host', 'Status', 'ID']
	rows = []
	peeks = []
	for info in running:
		try:
			row = [info.rname if 'rname' in info else (info.name if 'name' in info else f'{info.ID}.{info.proc}'),
			       info.date,
			       f'{info.done//1000}/{info.target//1000}', info.host if 'host' in info else 'N/A',
			       info.status, f'{info.ID}.{info.proc}']
		except Exception as e:
			print(dict(info))
			raise e
		rows.append(row)
		peeks.append(info.peek if 'peek' in info else None)
	title = 'Running jobs:'
	if show_peeks:
		print_separate_table(rows, cols, title, peeks)
	else:
		print_table(rows, cols, title)
	
	cols = ['Name', 'Date', 'Progress', 'Status']
	rows = []
	peeks = []
	for info in fail:
		if 'done' not in info:
			info.done = 0
		if 'target' not in info:
			info.target = '?'
		row = [info.rname if 'rname' in info else (info.name if 'name' in info else f'{info.ID}.{info.proc}'),
		       info.date,
		       f'{info.done//1000}/{info.target//1000 if isinstance(info.target, int) else info.target}',
		       info.status,
		       # info.error_msg if 'error_msg' in info else '[None]'
		       ]
		rows.append(row)
		peeks.append(info.peek if 'peek' in info else None)
	title = 'Failed jobs:'
	if show_peeks:
		print_separate_table(rows, cols, title, peeks)
	else:
		print_table(rows, cols, title)
	
	# summary
	
	print()
	print(tabulate([
		['Completed', len(success)],
		['Running', len(running)],
		['Failed', len(fail)],
	]))
	
	if list_failed:
		print('\n'.join(r.name for r in fail))
	

@Script('cls')
def get_status(path=None, homedir=None,
               load_configs=False, target=None, list_failed=False,
				skip_missing=False,
               since=None, last=5, peek=None):
	'''
	Script to get a status of the cluster jobs
	'''
	
	if peek is not None and peek <= 0:
		peek = None
	
	if path is None:
		assert 'FOUNDATION_SAVE_DIR' in os.environ, 'FOUNDATION_SAVE_DIR not set'
		path = os.environ['FOUNDATION_SAVE_DIR']
	save_dir = path
	del path
	
	on_cluster = os.path.isdir('/lustre/home/fleeb')
	
	if homedir is None:
		homedir = '/home/fleeb/jobs' if on_cluster else '/is/ei/fleeb/workspace/chome/jobs'
	
	jobs = load_registry(homedir, since=since, last=last)
	
	if on_cluster:
		print()
		current = collect_q_cmd()
		
		if current is not None:
			connect_current(jobs, current)
		
	if peek is not None and peek > 0:
		for info in jobs:
			if 'path' in info and 'proc' in info:
				opath = os.path.join(info.path, 'out{}.log'.format(info.proc))
				# print(opath)
				info.peek = peek_file(opath, peek)
	
	connect_saves(jobs, saveroot=save_dir, load_configs=load_configs)
	fill_status(jobs, target=target)
	
	print_status(jobs, list_failed=list_failed, show_peeks=peek, skip_missing=skip_missing)
	
	# print_current(current, simple=peek is None)
	
	
	
	return 0








# @Script('cls')
def get_old_status(peek=None):
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
	
	info.jnum, info.ID, info.proc = map(int, job.split('-'))
	
	info.date = datetime.strptime(date, '%y%m%d-%H%M%S')
	info.str_date = info.date.ctime()  # .strftime()

	return info

def old_collect_runs(path, recursive=False, since=None, last=5):
	
	if recursive:
		raise NotImplementedError
	
	runs = util.Table()
	
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
	
	runs.filter_self(lambda run: run.job in sel)
	
	return runs

def old_load_runs(runs, load_configs=False):
	
	for run in runs:
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
		

	
def print_run_status(runs, list_failed=False, peek=None):
	
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
	
	print_table(rows, cols, 'Completed jobs:')
	
	# running jobs
	
	cols = ['Name', 'Date', 'Progress', 'Status']
	
	rows = []
	for run in running:
		row = [run.name, run.date, f'{run.progress * 100:3.1f}', run.status]
		rows.append(row)
	
	print_table(rows, cols, 'Running jobs:')
	
	# failed jobs
	
	cols = ['Name', 'Date', 'Progress', 'Status']
	
	rows = []
	for run in fail:
		row = [run.name, run.date, f'{run.progress * 100:3.1f}', run.status]
		rows.append(row)
	
	print_table(rows, cols, 'Failed jobs:')
	
	
	# summary
	
	print()
	print(tabulate([
		['Completed', len(success)],
		['Running', len(running)],
		['Failed', len(fail)],
	]))
	
	if list_failed:
		print('/n'.join(r.name for r in fail))
	


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
		
		if current is not None:
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





