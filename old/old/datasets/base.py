import os
from pathlib import Path

from omnifig import script, component, creator, modifier

from omniplex import toy, Dataset, Datastream, HierarchyBuilder


from ..novo.base import DataProduct, DataBuilder
from ..novo.base import hparam, inherit_hparams, submodule, submachine, material, space, indicator, machine, Structured



class RootedDataset(Dataset, DataProduct):
	_dirname = None


	@hparam(inherit=True)
	def root(self):
		path = Path(os.getenv('OMNIDATA_PATH', 'local_data/')) / 'datasets'
		if self._dirname is not None:
			return path / self._dirname
		return path



class DownloadableDataset(RootedDataset):
	def is_downloaded(self):
		return self.root.exists()


	def download_data(self):
		raise NotImplementedError



class ImageClassificationData(DataBuilder, branch='image-classification'): pass



class ToyData(DataBuilder, branch='toy'): pass
ToyData.register_product('swiss-roll', toy.SwissRollDataset)
ToyData.register_product('helix', toy.HelixDataset, is_default=True)



class Manifolds(DataBuilder, branch='manifold'): pass
Manifolds.register_product('swiss-roll', toy.SwissRoll)
Manifolds.register_product('helix', toy.Helix, is_default=True)





