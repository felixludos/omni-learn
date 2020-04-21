
import sys, os, shutil
import subprocess
from tabulate import tabulate
import numpy as np

from .registry import Script

@Script('cstat', use_config=True)
def get_status(A):
	'''
	Script to get a status of the cluster jobs
	'''
	
	assert 'FOUNDATION_RUN_MODE' in os.environ and os.environ['FOUNDATION_RUN_MODE'] == 'cluster', 'Should be the cluster'
	
	print(subprocess.check_output(['ls', '-l']).decode())
	
	print('Getting job status...', end='')
	raw = subprocess.check_output(['condor_q', 'fleeb', '-af', 'jl,'])
	print(' received {} bytes'.format(len(raw)))
	
	print()
	
	if len(raw) == 0:
		print('No jobs running.')
	
	
	
	return 0

