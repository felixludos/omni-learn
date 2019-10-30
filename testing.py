import os, sys
import torch
import torchvision
import torch.multiprocessing as mp

import foundation as fd
from foundation import models
from foundation import util
from foundation import train

def t(x, i=None):
        print( x+1 if i is None else x+i)


if __name__ == '__main__':
        path = os.path.join(train.DEFAULT_DATA_PATH, 'emnist')
        torchvision.datasets.EMNIST(path, split='letters', download=True, train=True)

        # mp.freeze_support()
        # mp.set_start_method('spawn')
        # a = torch.arange(3)#.cuda()
        # p = mp.Process(target=t, args=(a,1,))
        # p.start()
        # p.join()
