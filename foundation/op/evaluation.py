
import sys, os
import tqdm

import torch

import omnifig as fig

from ..framework import Recordable, Schedulable, Evaluatable

from .. import util

from .loading import load_config, load_records, setup_logging, setup_records, \
	wrap_datasets, wrap_transaction
from .model import load_model
from .data import load_data


@fig.Script('eval')
def eval_model(A, model=None, datasets=None):
	
	if model is None or datasets is None:
		A = load_config(A)
	
	results = None
	
	identifier = A.pull('eval.identifier', 'eval')
	
	save_dir = A.pull('output.save_dir')
	
	results_path = os.path.join(save_dir, f'{identifier}.pth.tar')
	overwrite = A.pull('eval.overwrite', False)
	
	if os.path.isfile(results_path) and not overwrite:
		print('WARNING: will not overwrite results, so skipping evaluation')
	
	else:
		print('*' * 50)
		print(f'Evaluating trained model: {identifier} (overwrite={overwrite})')
		print('*' * 50)
		
		logger = setup_logging(A)
		
		if datasets is None:
			datasets = wrap_transaction(load_data, A.dataset)
			if not isinstance(datasets, tuple):
				datasets = datasets, None
		*trainsets, testset = datasets
		
		use_testset = A.pull('eval.use_testset', False)
		
		if use_testset and testset is None:
			print('Using testset')
			testset, = load_data(A, mode='test')
		else:
			print('Test dataset NOT used!')
			testset = None
		
		loaders = wrap_datasets(*trainsets, A=A)
		testloader = loaders[-1] if testset is None else wrap_datasets(testset, A=A)
		
		# use_best = A.pull('eval.use_best', False)
		# if use_best:
		# 	A.push('last', False)
		
		model = load_model(A)
		records = load_records(A)
		
		print('Loaded best model, trained for {} iterations'.format(records['total_steps']))
		
		records['training_steps'] = records['total_steps']
		
		logger.set_step(records['total_steps'])
		logger.set_tag_format('{}/{}'.format(identifier, '{}'))
		
		info = fig.get_config()
		info._A = A # full config, probably shouldnt be used
		info.A = A.eval # eval settings
		
		info.datasets = datasets
		info.loaders = loaders
		
		info.identifier = identifier
		info.logger = logger
		
		info.testset = testset
		info.testloader = testloader
		
		model.eval()
		results = model.evaluate(info)
		
		if results is not None:
			if overwrite:
				print('Results already found, but not overwriting.')
			
			torch.save(results, results_path)
			print(f'Results saved to {results_path}')
	
	return A, results, model, datasets





