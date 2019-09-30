
import sys, os
import numpy as np
import mujoco_py as mjp
from collections import namedtuple
from itertools import product

MY_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Mujoco_PR2(object):

	def __init__(self, model_file=None, mass=1):
		# print('starting')
		self.env_id = 'MjpPR2-v1'
		self.model_file = model_file if model_file is not None else os.path.join(MY_DIR_PATH,
		                                                                         'assets/pr2.xml')
		self.model = mjp.load_model_from_path(self.model_file)

		# print(1)
		self.sim = mjp.MjSim(self.model)

		assert self.sim.model.stat.extent == 1, 'set the extent to 1 in the xml'

		mn, mx = self.sim.model.jnt_range.T
		mn[self.sim.model.jnt_limited == 0] = 0
		mx[self.sim.model.jnt_limited == 0] = 2 * np.pi
		self.mn, self.mx = mn, mx

		self.viewer = None  # mjp.MjViewerBasic(self.sim)
		self.observation_space = namedtuple('o_s', ['shape', 'high', 'low'])(shape=(8,), high=np.array(
			[np.pi, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf]),
		                                                                     low=-np.array(
			                                                                     [np.pi, np.inf, np.pi, np.pi, np.pi,
			                                                                      np.inf, np.inf, np.inf]))
		self.action_space = namedtuple('a_s', ['shape', 'high', 'low'])(shape=(1,), high=np.array([1]),
		                                                                low=np.array([-1], ))
		self.spec = namedtuple('spec', ['timestep_limit', 'observation_dim', 'action_dim'])(5000,
		                                                                                    self.observation_space.shape[
			                                                                                    0],
		                                                                                    self.action_space.shape[0])
		self.horizon = self.spec.timestep_limit

		self.set_mass(mass)

	def depth_to_linear(self,
	                    depth):  # zNear and zFar values must set to 0.2 and 3.0 and model extent must be set to 1 in model xml
		zNear, zFar = self.sim.model.vis.map.znear, self.sim.model.vis.map.zfar
		return zFar * zNear / (zFar + depth * (zNear - zFar))

	def get_label_key(self):
		return dict(self.label_maker.get_key())

	def render(self, wd, ht, onscreen_render=False, show_label=False, show_depth=False):
		'''get images, view will render directly, imgs==True will return rgb and depth images.
		I think (view and imgs)==True doesn't work for some reason.'''
		# assert mode == 'rgb_array', 'Rendering must be offscreen, so mode must equal rgb_array'
		# width, height

		img_dim = wd, ht

		rgb = self.sim.render(*img_dim, camera_name='cam', depth=show_depth)
		if show_depth:
			rgb, depth = rgb
			depth = self.depth_to_linear(depth[::-1, :])  # transform to correct depth values
		rgb = rgb[::-1, :]
		# rgb = rgb[:,:,::-1]
		if not show_label:
			return (rgb, depth) if show_depth else rgb

		return (rgb, depth, self.label_maker.get_label(*img_dim)) if show_depth else (
			rgb, self.label_maker.get_label(*img_dim))

	def _get_obs(self):
		state = np.concatenate([self.sim.data.qpos.copy(), self.sim.data.qvel.copy()])#self.sim.data.sensordata[:-1].copy()
		# state[3] %= 2*np.pi
		return state

	def set_mass(self, mass):
		# print(self.sim.model.body_mass)
		self.sim.model.body_mass[1] = mass  # catcher
		# self.sim.model.body_mass[-1] = mass # object
		self.mass = mass

	def reset(self, state=None, trigger=None):
		''' reset environment randomly - should be called before beginning a new episode'''
		if state is None:
			# state = np.random.randn(8)
			# state[[0,2,3,4]] = np.random.rand(4)*2-1
			state = np.zeros(28)
			state[:14] = np.random.uniform(self.mn, self.mx)

			state[14:] = np.random.randn(14) * 0.3

		self.sim.data.qpos[:] = state[:14]
		self.sim.data.qvel[:] = state[14:]

		self.sim.forward()

		return self._get_obs()

	def step(self, action, n=10):
		''' take a step in the environment with the control 'action' '''

		action = max(min(action, 1), -1)

		self.sim.data.ctrl[0] = action
		for _ in range(n):
			self.sim.step()

		ob = self._get_obs()

		notdone = np.isfinite(ob).all()
		self.done = not notdone

		reward = float(self.sim.data.sensordata[-1] > 0)
		# reward = self.sim.data.sensordata[-1]

		return ob, reward, self.done, {}

	def get_body_com(self, body_name):
		return self.sim.data.get_body_xpos(body_name)




