import sys, os
from pathlib import Path
import omnifig as fig
from omnibelt import unspecified_argument

from .. import util

from .loading import respect_config

@fig.Script('eval', description='Evaluate an existing model')
def evaluate(A=None, run=None):
	'''
	Load and evaluate a model (by defaulting using the validation set)
	
	Use argument "use_testset" to evaluate on test set
	'''
	
	if A is None:
		assert run is not None, 'either run or A must not be None'
		A = run.get_config()
	else:
		respect_config(A)
		
	ret_run = False
	if run is None:
		ret_run = True
		assert A is not None, 'either run or A must not be None'
		run = fig.run('load-run', A)
	
	results = run.evaluate(config=A)
	
	ret_run = A.pull('ret_run', ret_run, silent=True)
	if ret_run:
		return results, run
	return results




