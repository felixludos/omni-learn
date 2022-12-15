from collections import OrderedDict
import numpy as np

from omnidata import data, toy, Buildable, hparam, inherit_hparams, spaces
from omnidata.data import flavors
# from omnidata.data import Noisy as _Noisy
# from omnidata import ces

from ..novo.base import DatasetBuilder



class DataStreams(DatasetBuilder, ident='stream', as_branch=True):
	pass


class SwissRoll(DataStreams, toy.SwissRoll):
	pass


class Helix(DataStreams, toy.Helix):
	pass



class ToyManifolds(DatasetBuilder, ident='manifold', as_branch=True):
	pass


class SwissRollDataset(ToyManifolds, toy.SwissRollDataset, ident='swiss-roll'):
	pass


class HelixDataset(ToyManifolds, toy.HelixDataset, ident='helix'):
	pass






