
import sys, os

try:
	from tqdm import tqdm
except ImportError:
	tqdm = None

from tabulate import tabulate

import humpack as hp

import omnifig as fig

FD_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(FD_PATH), 'trained_nets')

def collect_info(run):
	
	info = {
		'end': run.A['training']['step_limit'],
		'date': run.records['timestamp'],
		'steps': run.records['total_steps'],
		'val_loss': run.records['stats']['val'][-1]['loss']['avg']
					if 'val' in run.records['stats']
					   and len(run.records['stats']['val'])
					   and 'loss' in run.records['stats']['val']
					else None,
		
	}
	
	run.info = info
	
	# for k,v in info.items():
	# 	setattr(run, k, v)
	# return info
	#
	


# @fig.Script('report', description='compile a report of past runs')
def get_report(A):
	
	root = A.pull('saveroot', '<>root', None)
	if root is None:
		root = os.environ['OMNILEARN_SAVE_DIR'] if 'OMNILEARN_SAVE_DIR' in os.environ else DEFAULT_SAVE_PATH

	# print(root)

	if root is None:
		raise Exception('no saveroot found')
	
	pbar = A.pull('pbar', True)
	if tqdm is None:
		pbar = None
	
	names = os.listdir(root)
	
	# region Load Runs
	
	A.push('silent', True, overwrite=False)
	A.push('_type', A.pull('run_type', 'run'), overwrite=False)
	
	runs = hp.Table()
	
	print(f'Found {len(names)} runs')
	
	itr = tqdm(names) if pbar else iter(names)
	with A.silenced():
		for name in itr:
			
			C = fig.get_config()
			C.update(A)
			
			C.push('path', os.path.join(root, name))
			
			run = C.pull_self()
			
			runs.append(run)
		
	# endregion
	
	
	runs.map(collect_info)
	
	
	
	runs = sorted(runs, key=lambda run: run.info['date'], reverse=True)
	
	limit = A.pull('limit', None)
	
	if limit is not None:
		runs = runs[:limit]
	
	
	
	
	return runs





