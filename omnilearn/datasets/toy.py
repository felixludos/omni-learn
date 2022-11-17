from collections import OrderedDict
import numpy as np

from omnidata import data, toy, Buildable, hparam, inherit_hparams, spaces
# from omnidata.data import Noisy as _Noisy
# from omnidata import ces

from ..novo.base import DatasetBuilder


class SampledStream(data.Sampledstream, Buildable):
	n_samples = hparam(100, space=spaces.Naturals(), alias='n-samples')



class Noisy(toy.manifolds.Noisy, Buildable):
	noise_std = hparam(0., space=spaces.HalfBound(min=0.), alias='noise-std')



class ToyManifolds(DatasetBuilder, ident='manifold', as_branch=True):
	pass



class SwissRoll(toy.SimpleSwissRoll, ToyManifolds, ident='stream-swiss-roll'):
	Ax = hparam(np.pi / 2, space=spaces.Bound(min=0.))
	Ay = hparam(21., space=spaces.Bound(min=0.))
	Az = hparam(np.pi / 2, space=spaces.Bound(min=0.))

	freq = hparam(0.5, space=spaces.Bound(min=0.))
	tmin = hparam(3., space=spaces.Bound(min=0.))
	tmax = hparam(9., space=spaces.Bound(min=0.))



class Helix(toy.SimpleHelix, ToyManifolds, ident='stream-helix'):
	n_helix = hparam(2, space=spaces.Naturals(), alias='n-helix')

	periodic_strand = hparam(False, space=spaces.Binary())

	Rx = hparam(1., space=spaces.Bound(min=0.))
	Ry = hparam(1., space=spaces.Bound(min=0.))
	Rz = hparam(1., space=spaces.Bound(min=0.))

	w = hparam(1., space=spaces.Bound(min=0.))



@inherit_hparams('n_samples', 'Ax', 'Ay', 'Az', 'freq', 'tmin', 'tmax')
class SwissRollDataset(SampledStream, SwissRoll, ident='swiss-roll'):
	pass



@inherit_hparams('n_samples', 'n_helix', 'periodic_strand', 'Rx', 'Ry', 'Rz', 'w')
class HelixDataset(SampledStream, Helix, ident='helix'):
	pass








