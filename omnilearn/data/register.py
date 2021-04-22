
import omnifig as fig

from .. import util

class DatasetNotFoundError(Exception):
	pass

dataset_registry = util.Registry()

def dataset_registration(name, dataset):
	cmpn_name = f'dataset/{name}'
	fig.Component(cmpn_name)(dataset)
	
	dataset_registry.new(name, cmpn_name)


def register_dataset(name):
	def _reg_fn(fn):
		nonlocal name
		dataset_registration(name, fn)
		fn._dataset_ident = name
		return fn
	
	return _reg_fn


