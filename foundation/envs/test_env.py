import numpy as np
from .. import util
from .. import framework as fm
from . import general as gen


class Sum_Env(fm.Env):
    def __init__(self, dim=4, horizon=5):
        super(Sum_Env, self).__init__(spec=gen.EnvSpec(obs_space=gen.Continuous_Space((dim,), 1, 0),
                                                   act_space=gen.Continuous_Space((1,), dim, 0),
                                                   horizon=horizon),
                                      ID='Sum-v0')
    
    def seed(self, seed=None):
        pass
    
    def reset(self, init_state=None):
        obs = np.random.rand(4)
        self.goal = obs.sum()
        return obs
    
    def step(self, action):
        reward = -(action[0] - self.goal) ** 2
        obs = self.reset()
        
        return obs, reward, False, {}
    
    def render(self):
        print(self.goal)


class Sel_Env(fm.Env):
    def __init__(self, dim=4, horizon=5):
        super(Sel_Env, self).__init__(spec=gen.EnvSpec(obs_space=gen.Continuous_Space((dim,), 1, 0),
                                                   act_space=gen.Continuous_Space((1,), dim, 0),
                                                   horizon=horizon),
                                      ID='Sum-v0')
    
    def seed(self, seed=None):
        pass
    
    def reset(self, init_state=None):
        obs = np.random.rand(4)
        self.goal = obs[0]
        return obs
    
    def step(self, action):
        reward = -(action[0] - self.goal) ** 2
        obs = self.reset()
        
        return obs, reward, False, {}
    
    def render(self):
        print(self.goal)