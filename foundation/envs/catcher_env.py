
import sys, os
import numpy as np
import mujoco_py as mjp
from collections import namedtuple
from itertools import product

class Label_Maker:
	# TODO: allow for many colors by iteratively coloring 7 bodies at a time
	def __init__(self, sim, geom_groups, cam_name='external_camera_0'):
		self.sim = sim
		self.cam_name = cam_name
		
		self.geom_groups = geom_groups
		self._blank = np.zeros(4)
		self._blank[-1] = 1
		self.all_colors = np.array(list(product(*[[0,1]]*3, [1])))[1:][[0,1,3,2,4,5,6]] # order: b,g,r,c,m,y,w
		self.all_colors = self.all_colors[:3][::-1] # only use r,g,b
		assert len(self.all_colors) >= len(self.geom_groups), 'Not enough colors - max is ' + str(len(self._colors))

		self.colors = {geom:self._blank for geom in self.sim.model.geom_names}
		for idx, geoms in enumerate(self.geom_groups):
			for geom in geoms:
				assert geom in self.sim.model.geom_names, 'Error: couldnt find {} in geom names'.format(geom)
				self.colors[geom] = self.all_colors[idx]
				
		self.light_amb = self.sim.model.light_ambient.copy()
		self.light_dir = self.sim.model.light_directional.copy()
		
		self.settings = {}

		for gid, geom in enumerate(self.sim.model.geom_names):
			mid = self.sim.model.geom_matid[gid]
			self.settings[geom] = {
				'rgba': self.sim.model.geom_rgba[gid, :].copy(),
				'spec': self.sim.model.mat_specular[mid].copy(),
				'shiny': self.sim.model.mat_shininess[mid].copy(),
				'refl': self.sim.model.mat_reflectance[mid].copy(),
				'emission': self.sim.model.mat_emission[mid].copy(),
				'mat': self.sim.model.geom_matid[gid],
			}

	def get_key(self):
		return {i+1:geoms for i, geoms in enumerate(self.geom_groups)}

	def label(self):
		self.sim.model.light_ambient[:] = 1
		self.sim.model.light_directional[:] = 1
		for geom, color in self.colors.items():
			gid = self.sim.model.geom_name2id(geom)
			mid = self.sim.model.geom_matid[gid]
			self.sim.model.geom_rgba[gid, :] = color
			# self.sim.model.mat_specular[mid] = 0
			# self.sim.model.mat_shininess[mid] = 0
			# self.sim.model.mat_reflectance[mid] = 0
			self.sim.model.mat_emission[mid] = 1000.
			self.sim.model.geom_matid[gid] = 0

	def unlabel(self):
		self.sim.model.light_ambient[:] = self.light_amb
		self.sim.model.light_directional[:] = self.light_dir
		for geom, vals in self.settings.items():
			gid = self.sim.model.geom_name2id(geom)
			mid = vals['mat']
			# mid = self.sim.model.geom_matid[gid]
			self.sim.model.geom_rgba[gid, :] = vals['rgba'].copy()
			# self.sim.model.mat_specular[mid] = vals['spec']
			# self.sim.model.mat_shininess[mid] = vals['shiny']
			# self.sim.model.mat_reflectance[mid] = vals['refl']
			self.sim.model.mat_emission[mid] = vals['emission']
			self.sim.model.geom_matid[gid] = vals['mat']

	def get_label(self, wd, ht):
		self.label()
		raw_lbl = self.sim.render(wd, ht, camera_name=self.cam_name)
		raw_lbl = raw_lbl[::-1, :]
		self.unlabel()
		
		#return raw_lbl

		raw_lbl = (raw_lbl/255).astype(np.uint8)

		bg_sel = np.sum(raw_lbl, axis=2) == 0

		lbl = np.argmax(raw_lbl, axis=2) + 1
		lbl[bg_sel] = 0

		return lbl

MY_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Mujoco_Catcher(object):
	
	def __init__(self, model_file=None, mass=1):
		# print('starting')
		self.env_id = 'MjpCatcher-v1'
		self.model_file = model_file if model_file is not None else os.path.join(MY_DIR_PATH,'assets/catcher-black.xml')
		self.model = mjp.load_model_from_path(self.model_file)
		
		# print(1)
		self.sim = mjp.MjSim(self.model)
		
		assert self.sim.model.stat.extent == 1, 'set the extent to 1 in the xml'
		
		self.viewer = None  # mjp.MjViewerBasic(self.sim)
		self.observation_space = namedtuple('o_s', ['shape', 'high', 'low'])(shape=(8,), high=np.array(
			[np.pi, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf]),
		    low=-np.array([np.pi, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf]))
		self.action_space = namedtuple('a_s', ['shape', 'high', 'low'])(shape=(1,), high=np.array([1]),
																		low=np.array([-1], ))
		self.spec = namedtuple('spec', ['timestep_limit', 'observation_dim', 'action_dim'])(100,
																							self.observation_space.shape[
																								0],
																							self.action_space.shape[0])
		self.horizon = self.spec.timestep_limit
		self.label_maker = Label_Maker(self.sim, geom_groups=[{'gobj'}, {'b0', 'b1', 'b2'}, {'palm', 'finger'}], cam_name='cam')
		
		self.set_mass(mass)
		
	def depth_to_linear(self, depth):  # zNear and zFar values must set to 0.2 and 3.0 and model extent must be set to 1 in model xml
		zNear, zFar = self.sim.model.vis.map.znear, self.sim.model.vis.map.zfar
		return zFar * zNear / (zFar + depth * (zNear - zFar))
	
	def get_label_key(self):
		return dict(self.label_maker.get_key())
	
	def copy(self):
		return Mujoco_Catcher(self.model_file, self.mass)
	
	def render(self, mode='rgb_array', img_dim=(128, 128), onscreen_render=False, show_label=False, show_depth=False):
		'''get images, view will render directly, imgs==True will return rgb and depth images.
		I think (view and imgs)==True doesn't work for some reason.'''
		# assert mode == 'rgb_array', 'Rendering must be offscreen, so mode must equal rgb_array'
		# width, height
		
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
	
	def _get_obs(self):  # [cart_pos, pole_pos, cart_vel, pole_vel]
		state = self.sim.data.sensordata[:-1].copy()
		#state[3] %= 2*np.pi
		return state

	def set_mass(self, mass):
		#print(self.sim.model.body_mass)
		self.sim.model.body_mass[1] = mass # catcher
		#self.sim.model.body_mass[-1] = mass # object
		self.mass = mass
	
	def reset(self, state=None, trigger=None):
		''' reset environment randomly - should be called before beginning a new episode'''
		if state is None:
			#state = np.random.randn(8)
			#state[[0,2,3,4]] = np.random.rand(4)*2-1
			state = np.zeros(12)
			state[0] = np.random.rand()*1.5 - 0.75
			state[1] = np.random.randn() * 2

			state[6] = np.random.randn() * 0.05
			state[4] = np.random.randn() * 0.05

		self.sim.data.qpos[0] = state[0]
		self.sim.data.qvel[0] = state[1]
		self.sim.data.qpos[1:4] = state[2:5]
		self.sim.data.qvel[1:4] = state[5:8]
		self.sim.data.qpos[4:6] = state[8:10]
		self.sim.data.qvel[4:6] = state[10:12]

		if trigger is None:
			trigger = np.random.randint(140, 181)
		self.trigger = trigger

		self.sim.forward()

		self._cnt = 0

		return self._get_obs()

	def step(self, action, n=10):
		''' take a step in the environment with the control 'action' '''
		
		action = max(min(action,1),-1)
		
		self.sim.data.ctrl[0] = action
		for _ in range(n):
			self.sim.step()
			if self._cnt < self.trigger:
				self.sim.data.ctrl[1:] = 0.0
			else:
				self.sim.data.ctrl[1:] = -1.0
			self._cnt += 1
			
		ob = self._get_obs()
		
		notdone = np.isfinite(ob).all()
		self.done = not notdone

		reward = float(self.sim.data.sensordata[-1] > 0)
		#reward = self.sim.data.sensordata[-1]

		return ob, reward, self.done, {}

	def get_body_com(self, body_name):
		return self.sim.data.get_body_xpos(body_name)
