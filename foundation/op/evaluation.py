import omnifig as fig

from foundation.op.runs import NoOverwriteError
from .. import util

from .loading import respect_config

@fig.Script('eval', description='Evaluate an existing model')
def evaluate(A=None, run=None):
	'''
	Load and evaluate a model (by defaulting using the validation set)
	
	Use argument "use_testset" to evaluate on test set
	'''
	
	respect_config(A)
	
	ret_run = False
	if run is None:
		ret_run = True
		assert A is not None, 'either run or A must not be None'
		A.push('run._type', 'run', overwrite=False)
		run = A.pull('run')
	
	A = run.get_config()

	results = run.evaluate()
	
	ret_run = A.pull('ret_run', ret_run)
	if ret_run:
		return results, run
	return results




