
from omnibelt import get_printer
import omnifig as fig

prt = get_printer(__name__)


from ..data import dataset_registry, Downloadable

class DownloadError(Exception):
	pass

@fig.Script('download-dataset')
def download_dataset(A, **kwargs):
	
	name = A.pull('_dataset_type', '<>dataset-name', '<>name')
	
	cmpn_name = dataset_registry.get(name, None)
	if cmpn_name is None:
		raise DownloadError(f'Can\'t find dataset {name} (has it been registered?)')
	
	cmpn = fig.find_component(cmpn_name).fn
	if not issubclass(cmpn, Downloadable):
		raise DownloadError(f'{name} is not downloadable (it does not subclass `Downloadable`)')

	return cmpn.download(A, **kwargs)



