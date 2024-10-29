from .imports import *

from omniply.apps.training import Indexed


class DefaultPlanner(Indexed):
    def expected_samples(self, step_size: int) -> Optional[int]:
        total_itr = self.expected_iterations(step_size)
        if total_itr is None:
            return None
        return total_itr * step_size

