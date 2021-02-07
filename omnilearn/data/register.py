
import omnifig as fig

from .. import util


dataset_registry = util.Registry()

def register_dataset(name, dataset):
	cmpn_name = f'dataset/{name}'
	fig.Component(cmpn_name)(dataset)
	
	dataset_registry.new(name, cmpn_name)


def Dataset(name):
	def _reg_fn(fn):
		nonlocal name
		register_dataset(name, fn)
		fn._dataset_ident = name
		return fn
	
	return _reg_fn


