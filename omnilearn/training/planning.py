from .imports import *

from ..abstract import AbstractPlanner
from omniply.apps.training import Indexed



class DefaultPlanner(Indexed, AbstractPlanner):
    def expected_samples(self, step_size: int) -> Optional[int]:
        total_itr = self.expected_iterations(step_size)
        if total_itr is None:
            return None
        return total_itr * step_size


    def budget(self, *, max_samples: int = None, max_batches: int = None, 
               max_iterations: int = None, max_epochs: int = None, **kwargs):
        if max_samples is not None:
            self._max_samples = max_samples
        if max_batches is not None:
            self._max_batches = max_batches
        if max_iterations is not None:
            self._max_iterations = max_iterations
        if max_epochs is not None:
            self._max_epochs = max_epochs



