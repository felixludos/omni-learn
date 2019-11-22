
import sys, os
import numpy as np
import mujoco_py as mjp
from collections import namedtuple
from itertools import product
from .. import util
from ..sim import mujoco as mjutil

MY_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Mujoco_Object_PR2(object):

	def __init__(self, model_file=None, epsilon=1e-8,
				 xtrans=True, ytrans=True, ztrans=True,
				 xrot=True, yrot=True, zrot=True,
				 rand_on_reset=False, rot_pos_ctrl=None,
				 num_prim=3, mod_textures=True, mod_shape=True, mod_size=True, use_simple=False,
				 bg_files=None, tex_files=None,
				 bg_path='/mnt/wetlab_data/places365/data_large', tex_path='/mnt/wetlab_data/dtd/images/'):
		# print('starting')
		self.env_id = 'MjpPR2-v1'
		model_file = model_file if model_file is not None else os.path.join(MY_DIR_PATH,
		                                                                         'assets/pr2_left_obj.xml')

		num_obj = 1


		nodof = not xtrans, not ytrans, not ztrans, not xrot, not yrot, not zrot
		assert sum(nodof) < 6, 'must have at least 1 dof'
		self.dof = np.array([xtrans, ytrans, ztrans, xrot, yrot, zrot]).astype(int)

		assert mod_textures, 'must mod textures'
		if bg_files is None:
			bg_files = util.crawl(bg_path, lambda x: '.jpg' in x) if mod_textures else None
			print('bg files', len(bg_files))
		self.bg_files = bg_files
		if tex_files is None:
			tex_files = util.crawl(tex_path, lambda x: '.jpg' in x) if mod_textures else None
			print('tex files', len(tex_files))
		self.tex_files = tex_files
		self.mod_shape = mod_shape
		self.mod_size = mod_size
		self.mod_textures = mod_textures
		self.randomize_on_reset = rand_on_reset
		self.use_simple = use_simple

		with open(model_file, 'r') as f:
			model_tmpl = f.read()

		# self.neutral_pos = neutral_pos
		self.epsilon = epsilon
		self.num_obj = num_obj
		self.num_prim = num_prim

		bodies, acts, assets = mjutil.create_free_objs(1, self.num_prim, no_jnts=True)
		self.objnames = ['obj{}'.format(i) for i in range(self.num_obj)]

		self.model_str = model_tmpl.format(assets, bodies, acts)

		try:
			self.model = mjp.load_model_from_xml(self.model_str)
		except Exception as e:
			print(self.model_str, file=open('temp-mujoco.txt', 'w'))
			print('loading model failed saved model string')
			raise e
		self.randomize_env()
		# self.sim = mjp.MjSim(self.model)
		mn, mx = self.sim.model.jnt_range.T
		mn[self.sim.model.jnt_limited == 0] = 0
		mx[self.sim.model.jnt_limited == 0] = 2 * np.pi
		self.mn, self.mx = mn, mx

		self.neutral_pos = (self.mx - self.mn)/2 + self.mn


		nodof = (self.dof + 1) % 2

		self.viewer = None

		self.label_maker = mjutil.Label_Maker(self.sim, body_names=self.objnames, cam_name='cam')

		self.obs_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel)
		self.action_dim = len(self.sim.data.ctrl)
		self.horizon = 100

		self.zNear, self.zFar = self.sim.model.vis.map.znear, self.sim.model.vis.map.zfar

		self.cam_name = 'cam'

		if self.use_simple:
			self.simplify_env()

		# assert self.sim.model.stat.extent == 1, 'set the extent to 1 in the xml'

		# self.viewer = None  # mjp.MjViewerBasic(self.sim)

		# self.horizon = 200


	def set_camera(self, name):
		old = self.cam_name
		self.cam_name = name
		self.label_maker.cam_name = name
		return old

	def depth_to_linear(self,
	                    depth):  # zNear and zFar values must set to 0.2 and 3.0 and model extent must be set to 1 in model xml
		zNear, zFar = self.sim.model.vis.map.znear, self.sim.model.vis.map.zfar
		return zFar * zNear / (zFar + depth * (zNear - zFar))

	def get_label_key(self):
		return dict(self.label_maker.get_key())

	def randomize_env(self):
		self.sim = mjp.MjSim(self.model)
		self.obj_idx = self.sim.model.body_name2id('obj0')

		if self.mod_textures:
			mjutil.mod_textures(self.sim, self.tex_files, self.bg_files, bg_idx=0, use_floor=False)
			# tex = mjutil.pick_textures(self.sim, self.tex_files, self.bg_files, bg_idx=0)
			# mjutil.texture_geoms(self.sim, tex)

		if self.mod_shape or self.mod_size:
			for oname in self.objnames:
				mjutil.randomize_obj(self.sim, oname, mod_shape=self.mod_shape, mod_size=self.mod_size,
									 pos_mag=.03, size_mag=.05)

		self.sim.model.geom_size[self.sim.model.geom_bodyid == self.obj_idx] *= 0.8


	def render(self, wd, ht, show_label=False, show_depth=False):
		'''get images, view will render directly, imgs==True will return rgb and depth images.
		I think (view and imgs)==True doesn't work for some reason.'''
		# assert mode == 'rgb_array', 'Rendering must be offscreen, so mode must equal rgb_array'
		# width, height

		rgb = self.sim.render(wd, ht, camera_name=self.cam_name, depth=show_depth)
		if show_depth:
			rgb, depth = rgb
			depth = self.depth_to_linear(depth[::-1, :])  # transform to correct depth values
		rgb = rgb[::-1, :]
		# rgb = rgb[:,:,::-1]
		if not show_label:
			return (rgb, depth) if show_depth else rgb

		return (rgb, depth, self.label_maker.get_label(wd, ht)) if show_depth else (
			rgb, self.label_maker.get_label(wd, ht))

	def _get_obs(self):
		state = np.concatenate([self.sim.data.qpos.copy(), self.sim.data.qvel.copy()])#self.sim.data.sensordata[:-1].copy()
		# state[3] %= 2*np.pi
		return state

	def reset(self, state=None, trigger=None):
		''' reset environment randomly - should be called before beginning a new episode'''

		if self.randomize_on_reset and not self.use_simple:
			self.randomize_env()

		if state is None:
			# state = np.random.randn(8)
			# state[[0,2,3,4]] = np.random.rand(4)*2-1
			state = np.zeros(14)
			state[:7] = np.random.uniform(self.mn, self.mx)

			state[7:] = np.random.randn(7) * 0.3

		self.sim.data.qpos[:] = state[:7]
		self.sim.data.qvel[:] = state[7:]

		self.sim.forward()

		return self._get_obs()

	def step(self, action, n=10):
		''' take a step in the environment with the control 'action' '''

		action = action.clip(-1,1)

		self.sim.data.ctrl[:] = action
		for _ in range(n):
			self.sim.step()

		ob = self._get_obs()

		notdone = np.isfinite(ob).all()
		self.done = not notdone

		reward = 1#float(self.sim.data.sensordata[-1] > 0)
		# reward = self.sim.data.sensordata[-1]

		return ob, reward, self.done, {}

	def get_body_com(self, body_name):
		return self.sim.data.get_body_xpos(body_name)




