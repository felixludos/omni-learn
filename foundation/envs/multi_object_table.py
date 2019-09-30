
import sys, os
import numpy as np
import mujoco_py as mjp
from mujoco_py.modder import TextureModder
from itertools import product
from .. import util
from ..sim import mujoco as mjutil
from scipy.spatial.distance import pdist

MY_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Multi_Object_Table(object):

	def __init__(self, model_path=None, epsilon=1e-8,
				 xtrans=True, ytrans=True, ztrans=True,
				 xrot=True, yrot=True, zrot=True,
				 num_obj=1, rand_on_reset=True,
				 num_prim=3, mod_textures=True, mod_shape=True, mod_size=True, use_simple=False,
				 bg_path='/mnt/wetlab_data/places365/data_large', tex_path='/mnt/wetlab_data/dtd/images/'):

		nodof = not xtrans, not ytrans, not ztrans, not xrot, not yrot, not zrot
		assert sum(nodof) < 6, 'must have at least 1 dof'
		self.dof = np.array([xtrans, ytrans, ztrans, xrot, yrot, zrot]).astype(int)

		if model_path is None:
			# model_path = os.path.join(MY_DIR_PATH, 'assets', 'multi_object_full_template.xml')

			model_path = os.path.join(MY_DIR_PATH, 'assets', 'multi_object_table_template.xml')

		assert mod_textures, 'must mod textures'
		self.bg_files = util.crawl(bg_path, lambda x: '.jpg' in x) if mod_textures else None
		self.tex_files = util.crawl(tex_path, lambda x: '.jpg' in x) if mod_textures else None
		print('files', len(self.bg_files), len(self.tex_files))
		self.mod_shape = mod_shape
		self.mod_size = mod_size
		self.mod_textures = mod_textures
		self.randomize_on_reset = rand_on_reset
		self.use_simple = use_simple
		self.use_floor = True

		with open(model_path, 'r') as f:
			model_tmpl = f.read()

		# self.neutral_pos = neutral_pos
		self.epsilon = epsilon
		self.num_obj = num_obj
		self.num_prim = num_prim

		bodies, acts, assets = mjutil.create_free_objs(self.num_obj, self.num_prim,
		                                               pos_lim=.8, damping=.01, mass=4,
		                                               use_floor=self.use_floor, limit_z=1) #self.bg_files)
		self.objnames = ['obj{}'.format(i) for i in range(self.num_obj)]

		self.model_str = model_tmpl.format(assets, bodies, acts)

		# print(self.model_str)

		self.model = mjp.load_model_from_xml(self.model_str)
		self.randomize_env()
		# self.sim = mjp.MjSim(self.model)

		nodof = (self.dof + 1) % 2

		self.neutral_pos = np.zeros(6)
		self.ranges = []

		for oid in range(self.num_obj):

			# limit rot
			self.sim.model.jnt_limited[oid*6+3:(oid+1)*6] = nodof[3:]

			self.ranges.append(self.sim.model.jnt_range[oid*6:(oid+1)*6])
			for i, (nt, neutral) in enumerate(zip(nodof[:3], self.neutral_pos)):
				if nt:
					self.ranges[-1][i, 0] = neutral - self.epsilon
					self.ranges[-1][i, 1] = neutral + self.epsilon
			self.ranges[-1][3:, 1] = [self.epsilon if nr else 2 * np.pi for nr in nodof[3:]]
		self.ranges = np.stack(self.ranges)

		self.viewer = None

		self.label_maker = mjutil.Label_Maker(self.sim, body_names=self.objnames)

		self.obs_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel)
		self.action_dim = len(self.sim.data.ctrl)
		self.horizon = 100

		self.zNear, self.zFar = self.sim.model.vis.map.znear, self.sim.model.vis.map.zfar

		if self.use_simple:
			self.simplify_env()

	def seed(self, seed):
		np.random.seed(seed)

	def randomize_env(self):
		self.sim = mjp.MjSim(self.model)

		if self.mod_textures:
			mjutil.mod_textures(self.sim, self.tex_files, self.bg_files, bg_idx=0, use_floor=self.use_floor)
			# tex = mjutil.pick_textures(self.sim, self.tex_files, self.bg_files, bg_idx=0)
			# mjutil.texture_geoms(self.sim, tex)

		if self.mod_shape or self.mod_size:
			for oname in self.objnames:
				mjutil.randomize_obj(self.sim, oname, mod_shape=self.mod_shape, mod_size=self.mod_size,
									 shape_distrib=[1,1,1,1,8],
									 pos_mag=.06, size_mag=.07)

	def simplify_env(self):
		mjutil.set_simple_shape(self.sim)

	def sample_prior(self):

		pos = None

		while pos is None:

			pos = np.random.uniform(low=self.ranges[..., 0], high=self.ranges[..., 1]) # N x 6
			# order = np.arange(len(pos))
			#
			# pos[:,2] = order*0.5 + .3
			if self.num_obj > 1:
				dists = pdist(pos[:,:2])

				if dists.min() < .4:
					pos = None

		pos[:, 2] = np.random.uniform(0.25, 0.35)

		# print(pos.shape)
		# quit()


		# vel = np.zeros((self.num_obj, 6))
		vel = np.random.randn(self.num_obj, 6) * .5
		vel = vel.clip(-3, 3)
		vel[:, 3:] *= 1
		vel *= self.dof

		# vel *= 0



		return np.concatenate([pos.reshape(-1), vel.reshape(-1)],0)

	def reset(self, state=None):

		if self.randomize_on_reset and not self.use_simple:
			self.randomize_env()

		if state is None:
			state = self.sample_prior()

		# print(vel, self.rots)

		# else:
		assert len(state) == 12*self.num_obj
		pos = state[:6*self.num_obj]
		vel = state[6*self.num_obj:]

		self.sim.data.qpos[:] = pos.reshape(-1)
		self.sim.data.qvel[:] = vel.reshape(-1)

		self.sim.forward()

		#print('r')
		# self.step(n=200)

		return self._get_obs()

	def render(self, wd, ht, show_depth=False, show_label=False):

		rgb = self.sim.render(wd, ht, camera_name='external_camera_0', depth=show_depth)
		if show_depth:
			rgb, depth = rgb
			depth = mjutil.depth_to_linear(depth[::-1, :], self.zNear, self.zFar)  # transform to correct depth values
		rgb = rgb[::-1, :]
		if not show_label:
			return (rgb, depth) if show_depth else rgb

		return (rgb, depth, self.label_maker.get_label(wd, ht, self.sim)) if show_depth else (
		rgb, self.label_maker.get_label(wd, ht, self.sim))

	def _get_obs(self):
		state = np.concatenate([self.sim.data.qpos, self.sim.data.qvel], -1)#.reshape(2, self.num_obj, -1)  # euler
		# state[0,:,3:6] %= 2 * np.pi
		return state.reshape(-1)

	def step(self, action=None, n=1):

		if action is not None:
			# action = action.clip(-1,1)
			action = action.reshape(self.num_obj, -1) * self.dof
			# assert action.shape == (6,)
			self.sim.data.ctrl[:] = action.reshape(-1)

		for _ in range(n):
			self.sim.step()

		reward = 0
		obs = self._get_obs()

		return obs, reward, False, {}
