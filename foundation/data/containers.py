
import torch
from collections import OrderedDict

class Movable(object):

    def to(self, device):
        raise NotImplementedError

    def size(self, *args, **kwargs):
        raise NotImplementedError

class TensorDict(Movable, OrderedDict):

    def __init__(self, *args, _size_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._size_key = _size_key


    def to(self, device):
        for k,v in self.items():
            try:
                self[k] = v.to(device)
            except AttributeError:
                pass

        return self

    def _find_size_key(self):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self._size_key = k
                return

        raise Exception('No torch.Tensor found')

    def size(self, *args, **kwargs):
        if self._size_key is None:
            self._find_size_key()
        return self[self._size_key].size(*args, **kwargs)

class TensorList(Movable, list):

    def __init__(self, *args, _size_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._size_key = _size_key

    def to(self, device):
        for i,x in enumerate(self):
            try:
                self[i] = x.to(device)
            except AttributeError:
                pass

        return self

    def _find_size_key(self):
        for i, x in enumerate(self):
            if isinstance(x, torch.Tensor):
                self._size_key = i
                return

        raise Exception('No torch.Tensor found')

    def size(self, *args, **kwargs):
        if self._size_key is None:
            self._find_size_key()
        return self[self._size_key].size(*args, **kwargs)



