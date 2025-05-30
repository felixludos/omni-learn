from .imports import *
from .abstract import AbstractSelector
from ..abstract import AbstractDataset


class InfiniteSelector(AbstractSelector):
    def __init__(self, size: int = None, *, shuffle: bool = True, multi_epoch: bool = True,
                 seed: int = None, sort_indices: bool = True, **kwargs):
        if seed is None:
            seed = random.randint(1, 2 ** 32 - 1)
        if isinstance(size, AbstractDataset):
            size = size.size
        super().__init__(**kwargs)
        self._dataset_size = size
        self._num_iterations = 0
        self._shuffle = shuffle
        self._multi_epoch = multi_epoch
        self._sort_indices = sort_indices
        self._seed = seed
        self._initial_seed = seed
        self._order = None
        self._offset = 0
        self._drawn_epochs = 0
        self._drawn_samples = 0
        self._drawn_batches = 0

    def reset(self, size: int = None) -> Self:
        '''reset the selector state, optionally to a specific size'''
        if size is not None:
            self._dataset_size = size
        self._seed = self._initial_seed
        self._order = None
        self._offset = 0
        self._drawn_epochs = 0
        self._drawn_samples = 0
        self._drawn_batches = 0
        return self

    @staticmethod
    def _increment_seed(seed: int) -> int:
        '''deterministically change the seed'''
        return random.Random(seed).randint(1, 2 ** 32 - 1)

    def _draw_indices(self, n: int):

        if self._dataset_size is None:
            return None

        if self._order is None:
            if self._drawn_samples > 0:
                self._seed = self._increment_seed(self._seed)
            self._order = np.random.RandomState(self._seed).permutation(
                self._dataset_size) if self._shuffle else np.arange(self._dataset_size)
            self._offset = 0
            self._drawn_epochs += 1

        assert self._multi_epoch or n < len(self._order), f'batch size is too large: max is {len(self._order)}'

        if self._offset + n > len(self._order):  # need to wrap around
            indices = self._order[self._offset:]
            self._order = None
            if self._multi_epoch:
                indices = np.concatenate((indices, self._draw_indices(n - len(indices))))
        else:
            indices = self._order[self._offset:self._offset + n]
            self._offset += n

        if self._sort_indices:
            indices.sort()
        return indices

    def draw(self, n: int, /) -> np.ndarray:
        assert n > 0, 'cannot draw zero samples'  # otherwise some batches can have degenerate seeds
        self._drawn_batches += 1
        self._drawn_samples += n
        return self._draw_indices(n)


