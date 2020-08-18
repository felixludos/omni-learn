
import sys, os
import subprocess

from datetime import datetime

import omnibelt as belt
import humpack as hp

import omnifig as fig


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
		info = hp.adict()
	
	info.jnum = num
	# info.raw_date = date
	info.date = belt.get_now()
	# info.str_date = info.date.ctime()#.strftime()
	
	info.jexe = jexe
	info.path = os.path.dirname(raw)
	
	return info


def parse_remotehost(raw):
	s, g = raw.split('.')[0].split('@')
	s = s.split('_')[-1]
	return f'{s}{g}'


def parse_job_status(raw):
	info = hp.adict()
	
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
	print('Getting job status ... ', end='')
	
	try:
	
		# raw = subprocess.check_output(['condor_q', 'fleeb', '-af', 'ClusterId', 'ProcId', 'Args', 'JobStatus', 'RemoteHost', 'Env'])
		raw = subprocess.check_output(['condor_q', 'fleeb', '-af:t'] + colattrs).decode()
	
	except FileNotFoundError:
		print('FAILED')
		return None
	
	if len(raw) == 0:
		print('No jobs running.')
		return None
	else:
		lines = raw.split('\n')
		print('found {} jobs'.format(len(lines) - 1))
	
	# print(lines)
	
	active = hp.Table(parse_job_status(hp.adict(zip(colattrs, line.split('\t')))) for line in lines if len(line))
	
	return active


@fig.Script('status', description='check the status of jobs submitted to the cluster')
def get_status(A):
	
	collect_q_cmd()
	
	pass

