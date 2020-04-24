
import sys, os, shutil
import subprocess
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
	info.date = date
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
		info.proc_id = info.ProcId
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
			del info.RemoteHost
		except Exception:
			pass
	
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
	
	print(lines)
	
	R = util.MultiDict(parse_job_status(util.tdict(zip(colattrs, line.split('\t')))) for line in lines)
	
	return R

def peek_file(opath, peek=0):
	if os.path.isfile(opath):
		with open(opath, 'r') as f:
			return f.readlines()[-peek:]
	return None

def print_current(full, simple=True):
	table = []
	for info in full:
		table.append([f'{info.job_id}.{info.proc_id}', f'{info.num}-{info.proc_id}', f'{info.date}', f'{info.status}'])
		
	if simple:
		print(tabulate(table, ['ClusterId', 'JobId', 'StartDate', 'Status']))
	
	else:
		for row, info in zip(table, full):
			head = '----- {} -----'.format(' - '.join(row))
			tail = '-'*len(head)
			
			print(head)
			print(info.peek)
			print(tail)
			print()

			
@Script('cstat')
def get_status(peek=None):
	'''
	Script to get a status of the cluster jobs
	'''
	
	# assert 'FOUNDATION_RUN_MODE' in os.environ and os.environ['FOUNDATION_RUN_MODE'] == 'cluster', 'Should be the cluster'
	
	print()
	
	current = collect_q_cmd()
	
	print(current)
	
	if current is None or len(current) == 0:
		# print('No jobs running.')
		return 0
	
	if peek is not None:
		for info in current:
			if 'path' in info and 'num' in info:
				opath = os.path.join(info.path, 'out{}.log'.format(info.num))
				info.peek = peek_file(opath, peek)
	
	print_current(current, simple=peek is None)
	
	return 0

