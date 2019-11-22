import numpy as np
from collections import deque
from lab_setup import *

class Stat(object):
    
    def __init__(self):
        self.name = 'empty_stat'
        self.title = 'Empty Stat'
        self.level = None
        self.labels = None
        self._data = deque()
        self.y_axis = None
    
    def reset(self, env):
        self._data = deque()
        
    def calc(self, env):
        raise NotImplementedError('This function should be overwritten by the stat')
        
    def get_data(self):
        return self._data
    
    def __str__(self):
        return '<' + self.title + ' Stat Object>'
        
class Energy(Stat):
    def __init__(self):
        super(Energy, self).__init__()
        self.name = 'energy'
        self.title = 'Energy'
        self.labels = ['KE', 'PE', 'TE']
        self.y_axis = 'Energy (J)'
        
    def calc(self, env):
        # use multithreading here
        for action in env.forces:
            action.calc_energy(env)
        U = np.sum([action.energy for action in env.forces])
        T = 0.5 * np.sum(env.masses * np.sum(env.vel**2, axis=1, keepdims=True))
        self._data.append((T, U, T + U))
        
    def get_data(self):
        self.level = self._data[0][2]
        return np.vstack(self._data).T

class Temperature(Stat):
    def __init__(self):
        super(Temperature, self).__init__()
        self.name = 'temp'
        self.title = 'Temperature'
        self.y_axis = 'Temperature (K)'
        
    def reset(self, env):
        self.level = np.mean(env.natural_line_widths) * h_bar / kB / 2
        super(Temperature, self).reset(env)
        
    def calc(self, env):
        avg_sqr_vel = np.mean(np.sum(env.masses*env.vel**2,axis=1))
        self._data.append(avg_sqr_vel / 3 / kB)
    
    def get_data(self):
        return np.array([self._data])
