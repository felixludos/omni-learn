from .imports import *

from omniply.apps.training import Indexed



class DefaultPlanner(Indexed):
    '''Note: indices are numpy arrays'''
    def _draw_indices(self, n: int):
        indices = super()._draw_indices(n)
        indices.sort() # requirement for h5py read by index list
        return indices

