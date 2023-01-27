from collections import OrderedDict
import numpy as np

from omnidata import toy, Buildable, hparam, inherit_hparams, spaces

from ..novo.base import DataProduct, DataBuilder



class DataStreams(DataBuilder, ident='stream', as_branch=True):
	pass


@inherit_hparams('Ax', 'Ay', 'Az', 'freq', 'tmin', 'tmax')
class SwissRoll(DataProduct, toy.SwissRoll, registry=DataStreams, ident='swissroll'):
	pass


@inherit_hparams('n_helix', 'periodic_strand', 'Rx', 'Ry', 'Rz', 'w')
class Helix(DataProduct, toy.Helix, registry=DataStreams, ident='helix'):
	pass



class ToyManifolds(DataBuilder, ident='manifold', as_branch=True):
	pass


@inherit_hparams('n_samples', 'Ax', 'Ay', 'Az', 'freq', 'tmin', 'tmax')
class SwissRollDataset(DataProduct, toy.SwissRollDataset, registry=ToyManifolds, ident='swissroll'):
	pass


@inherit_hparams('n_samples', 'n_helix', 'periodic_strand', 'Rx', 'Ry', 'Rz', 'w')
class HelixDataset(DataProduct, toy.HelixDataset, registry=ToyManifolds, ident='helix'):
	pass






