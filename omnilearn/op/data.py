
import os

import numpy as np
import torch

import omnifig as fig

# from ..data import get_loaders, Info_Dataset, Subset_Dataset, simple_split_dataset, DataLoader, BatchedDataLoader
from .. import util


# FD_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# print(FD_PATH)


@fig.Modification('subset')
def make_subset(dataset, info):
	num = info.pull('num', None)
	
	shuffle = info.pull('shuffle', True)
	
	if num is None or num == len(dataset):
		print('WARNING: no subset provided, using original dataset')
		return dataset
	
	assert num <= len(dataset), '{} vs {}'.format(num, len(dataset))
	
	inds = torch.randperm(len(dataset))[:num].numpy() if shuffle else np.arange(num)
	return Subset_Dataset(dataset, inds)


