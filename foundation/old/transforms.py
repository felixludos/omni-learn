
import sys, os
import torch
import numpy as np
from collections import deque
import scipy.misc

def get_shuffle_setting(key='reflect'):
    if key == 'reflect':
        return reflect
    else:
        raise Exception('Unknown key: {}'.format(key))

#####################
# discrete
#####################

# grid
def rotate(obs): # A x L x L x 2
    pass

def reflect(obs, axis=None): # A x L x L x 2
    if axis is None:
        axis = np.randint(3, size=obs.shape[0])

    assert len(axis) == obs.shape[0]

    new_obs = obs.copy()
    new_obs[axis==1] = new_obs[axis==1, ::-1]
    new_obs[axis==2] = new_obs[axis==2, :, ::-1]

    return new_obs, axis

# coords

#####################
# continuous
#####################

# grid

# coords
