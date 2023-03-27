import os
from pathlib import Path

from omnifig import script, component, creator, modifier

from omnidata import hparam, inherit_hparams, submodule, submachine, spaces, material, space, indicator, machine, \
	Structured
from omnidata import Dataset, Datastream, HierarchyBuilder



class RootedDataset(Dataset):
	_dirname = None

	@hparam(inherit=True)
	def root(self):
		path = Path(os.getenv('OMNIDATA_PATH', 'local_data/')) / 'datasets'
		if self._dirname is not None:
			return path / self._dirname
		return path




class Data(HierarchyBuilder):
	pass



class Toy(Data, branch='toy', default_ident='swiss-roll', products={
	'swiss-roll': toy.SwissRollDataset,
	'helix': toy.HelixDataset,
	}):
	pass



class Manifolds(Data, branch='manifold', default_ident='swiss-roll', products={
	'swiss-roll': toy.SwissRoll,
	'helix': toy.Helix,
	}):
	pass






