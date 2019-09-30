
import sys, os
import numpy as np
import mujoco_py as mjp
from itertools import product

MY_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Label_Maker:
	# TODO: allow for many colors by iteratively coloring 7 bodies at a time
	def __init__(self, sim, body_names=None, cam_name='external_camera_0'):
		self.sim = sim
		self.cam_name = cam_name
		if body_names is None:
			body_names = {name for name in self.sim.model.body_names if name != 'world' and 'camera' not in name}
			print('WARNING: automatically choosing bodies to label: {}'.format(body_names))
		else:
			for name in body_names:
				assert name in self.sim.model.body_names, 'Error: couldnt find {} in body names'.format(name)
		self.body_names = list(body_names) # ordered
		#a = np.arange(0, 2)
		#self._colors = np.array(list(product(a,a,a)))[1:, :] # don't include black
		#self._colors = self._colors / 255.
		self._blank = np.zeros(4)
		self._colors = np.array(list(product(*[[0,1]]*3, [1])))[1:][[0,1,3,2,4,5,6]] # order: b,g,r,c,m,y,w
		self._colors = self._colors[:3:-1] # only use r,g,b
		assert len(self._colors) >= len(self.body_names), 'Not enough colors - max is ' + str(len(self._colors))

		self._colors = {body:c for body, c in zip(self.body_names, self._colors)}
		self.label_settings = {body:{} for body in self.sim.model.body_names}

		#print(self.sim.model.mat_emission)
		#print(self.label_settings)

		for gid, geom_body in enumerate(self.sim.model.geom_bodyid):
			body = self.sim.model.body_names[geom_body]
			if body in self.label_settings:
				mid = self.sim.model.geom_matid[gid]
				self.label_settings[body][gid] = {
					'rgba': self.sim.model.geom_rgba[gid, :].copy(),
					'spec': self.sim.model.mat_specular[mid].copy(),
					'shiny': self.sim.model.mat_shininess[mid].copy(),
					'refl': self.sim.model.mat_reflectance[mid].copy(),
					'emission': self.sim.model.mat_emission[mid].copy()
				}

	def get_key(self):
		return {name:i+1 for i, name in enumerate(self.body_names)}

	def label(self):
		for body in self.sim.model.body_names:
			color = self._colors[body] if body in self._colors else self._blank
			for gid in self.label_settings[body]:
				mid = self.sim.model.geom_matid[gid]
				self.sim.model.geom_rgba[gid, :] = color  # 1. # whiten materials
				# self.sim.model.mat_rgba[mat_id, :] = self._colors[i][:] # 1. # whiten materials
				# self.sim.model.tex_rgb[tex_id*3:(tex_id+1)*3] = (self._colors[i][:-1]*255).astype('uint8')
				self.sim.model.mat_specular[mid] = 0
				self.sim.model.mat_shininess[mid] = 0
				self.sim.model.mat_reflectance[mid] = 0
				self.sim.model.mat_emission[mid] = 1000.

	def unlabel(self):
		for body, gids in self.label_settings.items():
			for gid, vals in gids.items():
				mid = self.sim.model.geom_matid[gid]
				self.sim.model.geom_rgba[gid, :] = vals['rgba'].copy()
				self.sim.model.mat_specular[mid] = vals['spec']
				self.sim.model.mat_shininess[mid] = vals['shiny']
				self.sim.model.mat_reflectance[mid] = vals['refl']
				self.sim.model.mat_emission[mid] = vals['emission']

	def get_label(self, wd, ht):
		self.label()
		raw_lbl = self.sim.render(wd, ht, camera_name=self.cam_name)
		raw_lbl = raw_lbl[::-1, :]
		self.unlabel()

		#return raw_lbl

		bg_sel = np.sum(raw_lbl, axis=2) == 0

		lbl = np.argmax(raw_lbl, axis=2) + 1
		lbl[bg_sel] = 0

		return lbl

def depth_to_linear(depth, zNear, zFar):
	return zFar*zNear / (zFar + depth * (zNear - zFar))

class Free_Mass:

	def __init__(self, model_path=None, neutral_pos=[0.4,0,0.2],epsilon = 1e-8,
				 xtrans=True, ytrans=True, ztrans=True,
				 xrot=True, yrot=True, zrot=True):
		
		nodof = not xtrans, not ytrans, not ztrans, not xrot, not yrot, not zrot
		assert sum(nodof) < 6, 'must have at least 1 dof'
		self.dof = np.array([xtrans, ytrans, ztrans, xrot, yrot, zrot]).astype(int)
		
		if model_path is None:
			model_path = os.path.join(MY_DIR_PATH, 'assets', 'free_mass_dof.xml')

		self.model = mjp.load_model_from_path(model_path)
		
		self.sim = mjp.MjSim(self.model)

		# limit rot
		self.sim.model.jnt_limited[3:] = np.array(nodof[3:]).astype(int)
		
		self.ranges = self.sim.model.jnt_range
		for i, (nt, neutral) in enumerate(zip(nodof[:3], neutral_pos)):
			if nt:
				self.ranges[i, 0] = neutral - epsilon
				self.ranges[i, 1] = neutral + epsilon
		self.ranges[3:, 1] = [epsilon if nr else 2*np.pi for nr in nodof[3:]]
		
		self.viewer = None
		self.obs_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel)
		self.action_dim = len(self.sim.data.ctrl)
		self.horizon = 100

		self.zNear, self.zFar = self.sim.model.vis.map.znear, self.sim.model.vis.map.zfar

		self.label_maker = Label_Maker(self.sim, body_names=['floating_obj'])


	def reset(self, state=None):

		if state is None:

			pos = np.random.uniform(low=self.ranges[:,0], high=self.ranges[:,1])

			vel = np.random.randn(6) * 0.5
			vel = vel.clip(-2,2)
			vel[3:] *= 3
			vel *= self.dof
			#print(vel, self.rots)
			
		else:
			assert len(state) == 12
			pos = state[:6]
			vel = state[6:]

		self.sim.data.qpos[:] = pos
		self.sim.data.qvel[:] = vel

		self.sim.forward()

		return self._get_obs()


	def render(self, wd, ht, show_depth=False, show_label=False):

		rgb = self.sim.render(wd, ht, camera_name ='external_camera_0', depth=show_depth)
		if show_depth:
			rgb, depth = rgb
			depth = depth_to_linear(depth[::-1, :], self.zNear, self.zFar)  # transform to correct depth values
		rgb = rgb[::-1,:]
		if not show_label:
			return (rgb, depth) if show_depth else rgb

		return (rgb, depth, self.label_maker.get_label(wd, ht)) if show_depth else (rgb, self.label_maker.get_label(wd, ht))

	def _get_obs(self):
		state = np.concatenate([self.sim.data.qpos, self.sim.data.qvel], -1) # euler
		state[3:6] %= 2*np.pi
		return state

	def step(self, action, n=1):

		action = action.clip(-1, 1) * self.dof
		#assert action.shape == (6,)
		self.sim.data.ctrl[:] = action

		for _ in range(n):
			self.sim.step()

		reward = 0
		obs = self._get_obs()

		return obs, reward, False, {}







