import numpy as np
from gym import utils
from .mujoco_env import MujocoEnv
from mujoco_py import MjViewer
import os

class SimpleAntEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        MujocoEnv.__init__(self, os.path.join(curr_dir,'assets', 'ant.xml'), 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        xposbefore = self.data.body_xpos[1,0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.body_xpos[1,0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .1 * np.square(a).sum() # was 0.1
        reward = forward_reward - 0*ctrl_cost
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(forward_reward=forward_reward, ctrl_cost=ctrl_cost)

    def _get_obs(self):
        return np.concatenate([

            self.data.sensordata.copy(),
            self.data.qpos[2:7],
            self.data.qvel[:7]
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent*1.2

class VerySimpleAntEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        MujocoEnv.__init__(self, os.path.join(curr_dir,'assets', 'ant.xml'), 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        xposbefore = self.data.body_xpos[1,0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.body_xpos[1,0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .1 * np.square(a).sum()
        reward = forward_reward - ctrl_cost
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(forward_reward=forward_reward, ctrl_cost=ctrl_cost)

    def _get_obs(self):
        return np.concatenate([
            self.data.sensordata.copy(),
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent*1.2


class AntEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        MujocoEnv.__init__(self, os.path.join(curr_dir,'assets', 'ant.xml'), 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        xposbefore = self.data.body_xpos[1,0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.body_xpos[1,0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .1 * np.square(a).sum()
        reward = forward_reward - ctrl_cost
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(forward_reward=forward_reward, ctrl_cost=ctrl_cost)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent*1.2
