from .imports import *
import omnifig as fig

from .datasets import DatasetBase

# configurable versions of the top level functions


class Dataset(fig.Configurable, DatasetBase):
    pass
