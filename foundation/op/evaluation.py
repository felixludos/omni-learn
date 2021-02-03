import sys, os
from pathlib import Path
import omnifig as fig
from omnibelt import unspecified_argument

from .runs import NoOverwriteError
from .. import util

from .loading import respect_config

@fig.Script('eval', description='Evaluate an existing model')
def evaluate(A=None, run=None):
	'''
	Load and evaluate a model (by defaulting using the validation set)
	
	Use argument "use_testset" to evaluate on test set
	'''
	
	if A is not None:
		respect_config(A)
		
	
	ret_run = False
	if run is None:
		ret_run = True
		
		assert A is not None, 'either run or A must not be None'
		
		override = A.pull('override', None, raw=True, silent=True)
		
		name = A.pull('name', '<>path', '<>load', '<>resume')
		path = Path(name)
		
		if not path.is_dir():
			saveroot = A.pull('saveroot', os.environ.get('FOUNDATION_SAVE_DIR', '.'))
			path = saveroot / path
		
		assert path.is_dir(), f'run: {name} not found'
		
		config = fig.get_config(str(path))
		config.push('path', name)
		config.push('saveroot', saveroot)
		if override is not None:
			config.update({'override': override})
		run = config.pull('run')
	
	# A = run.get_config()
	
	results = run.evaluate(config=A)
	
	ret_run = A.pull('ret_run', ret_run, silent=True)
	if ret_run:
		return results, run
	return results




