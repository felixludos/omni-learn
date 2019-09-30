import sys
import os
#import matplotlib.pyplot as plt
import numpy as np
import mujoco_py as mjp
from collections import namedtuple
from itertools import product
from datetime import datetime
from PIL import Image

#import matplotlib.pyplot as plt

# TODO: fix model path
CARTPOLE_DIR_PATH = ''

class Label_Maker:

	def __init__(self, sim):
		self.sim = sim
		#a = np.arange(0, 2)
		#self._colors = np.array(list(product(a,a,a)))[1:, :] # don't include black
		#self._colors = self._colors / 255.
		self._colors = np.eye(3)
		self._colors = np.vstack([self._colors.T, np.ones(len(self._colors))]).T  # make them rgba
		assert len(self._colors) >= len(self.sim.model.geom_names), 'Not enough colors - max is ' + str(len(self._colors))
		self.label_key = {name: i + 1 for i, name in enumerate(self.sim.model.geom_names)}
		self.label_settings = {prop: [] for prop in ['rgba', 'spec', 'shiny', 'refl', 'emission']}

		for name in self.sim.model.geom_names:
			geom_id = self.sim.model.geom_name2id(name)
			mat_id = self.sim.model.geom_matid[geom_id]
			# print(self.sim.model.mat_rgba)
			# print(self.sim.model.mat_specular)
			self.label_settings['rgba'].append(self.sim.model.geom_rgba[geom_id, :].copy())
			# self.label_settings['rgba'].append(self.sim.model.mat_rgba[mat_id, :].copy())
			self.label_settings['spec'].append(self.sim.model.mat_specular[mat_id].copy())
			self.label_settings['shiny'].append(self.sim.model.mat_shininess[mat_id].copy())
			self.label_settings['refl'].append(self.sim.model.mat_reflectance[mat_id].copy())
			self.label_settings['emission'].append(self.sim.model.mat_emission[mat_id].copy())

	def get_key(self):
		return self.label_key

	def _set_label_settings(self, enable):
		if enable:
			for i, name in enumerate(self.sim.model.geom_names):
				geom_id = self.sim.model.geom_name2id(name)
				mat_id = self.sim.model.geom_matid[geom_id]
				#tex_id = self.sim.model.mat_texid[mat_id]
				self.sim.model.geom_rgba[geom_id, :] = self._colors[i][:]  # 1. # whiten materials
				#self.sim.model.mat_rgba[mat_id, :] = self._colors[i][:] # 1. # whiten materials
				#self.sim.model.tex_rgb[tex_id*3:(tex_id+1)*3] = (self._colors[i][:-1]*255).astype('uint8')
				self.sim.model.mat_specular[mat_id] = 0
				self.sim.model.mat_shininess[mat_id] = 0
				self.sim.model.mat_reflectance[mat_id] = 0
				self.sim.model.mat_emission[mat_id] = 1000.
			return
		for i, name in enumerate(self.sim.model.geom_names):
			geom_id = self.sim.model.geom_name2id(name)
			mat_id = self.sim.model.geom_matid[geom_id]
			#tex_id = self.sim.model.mat_texid[mat_id]
			self.sim.model.geom_rgba[geom_id, :] = self.label_settings['rgba'][i][:]
			#self.sim.model.mat_rgba[mat_id, :] = self.label_settings['rgba'][i][:]
			#self.sim.model.tex_rgb[tex_id * 3:(tex_id + 1) * 3] = (self.label_settings['rgba'][i][:-1] * 255).astype('uint8')
			self.sim.model.mat_specular[mat_id] = self.label_settings['spec'][i]
			self.sim.model.mat_shininess[mat_id] = self.label_settings['shiny'][i]
			self.sim.model.mat_reflectance[mat_id] = self.label_settings['refl'][i]
			self.sim.model.mat_emission[mat_id] = self.label_settings['emission'][i]

	def get_label(self, img_dim):
		self._set_label_settings(True)
		raw_lbl = self.sim.render(*img_dim, camera_name='cam')
		raw_lbl = raw_lbl[::-1, :]
		self._set_label_settings(False)

		#return raw_lbl

		bg_sel = np.sum(raw_lbl, axis=2) == 0

		lbl = np.argmax(raw_lbl, axis=2) + 1
		lbl[bg_sel] = 0

		return lbl


class Mujoco_Pendulum(object):
	'''
	Random start positions, so agent must learn to invert pendulum as well as keeping it inverted.
	The sigma parameter is used to make training easier: 
	for 0<sigma<pi the pendulum's initial position will be sampled from N(0,sigma),
	and sigma will be incremented by 'increment' after each episode.
	For any other sigma, the initial position will be chosen uniformly at random between -pi and pi.
	This should help training, because the first few episodes should train the agent the learn how to keep the pendulum inverted, and only then train the agent to learn how to invert the pendulum.
	'''
	def __init__(self, sigma=0, increment=0.002, limit=1.8, limited=False, pbc=False): # before: sigma=0.001
		#print('starting')
		curr_dir = os.path.dirname(os.path.abspath(__file__))
		model_file = 'inverted_pendulum_rgb_limited.xml' if limited else 'inverted_pendulum_rgb_unlimited.xml'
		path = os.path.join(curr_dir, 'assets', model_file)
		self.env_id = 'MjpInvertedPendulum-v1'
		self.model_file = model_file
		assert limited
		self.limited = limited
		self.model = mjp.load_model_from_path(path)

		if limited or pbc:
			limit = 1.2
		self.keep_in_view = limit
		self.pbc = pbc

		#print(1)
		self.sim = mjp.MjSim(self.model)
		self.viewer = None #mjp.MjViewerBasic(self.sim)
		self.observation_space = namedtuple('o_s', ['shape', 'high', 'low'])(shape=(4,), high=np.array([limit, np.pi, np.inf, np.inf]), low=np.array([-limit, -np.pi, -np.inf, -np.inf]))
		self.action_space = namedtuple('a_s', ['shape', 'high', 'low'])(shape=(1,), high=np.array([3]), low=np.array([-3], ))
		self.spec = namedtuple('spec', ['timestep_limit', 'observation_dim', 'action_dim'])(200, self.observation_space.shape[0], self.action_space.shape[0])
		self.horizon = self.spec.timestep_limit
		self.increment = increment

		assert not (limited and pbc), 'cant be limited with pbc'
		assert not limited or limit, 'must set a limit when using limited'

		self.label_maker = Label_Maker(self.sim)
		if not sigma:
			self.sigma = np.pi
		else:
			self.sigma = sigma

	def depth_to_linear(self, depth): # zNear and zFar values must set to 0.2 and 3.0 and model extent must be set to 1 in model xml
		zNear, zFar = 0.2, 3.0
		return zFar*zNear / (zFar + depth * (zNear - zFar))

	def get_label_key(self):
		return dict(self.label_maker.get_key())

	def copy(self):
		return Mujoco_Pendulum(self.model_file, self.sigma, self.increment, self.keep_in_view)

	def render(self, mode='rgb_array', img_dim=(320,160), onscreen_render=False, show_label=False, show_depth=False):
		'''get images, view will render directly, imgs==True will return rgb and depth images.
		I think (view and imgs)==True doesn't work for some reason.'''
		#assert mode == 'rgb_array', 'Rendering must be offscreen, so mode must equal rgb_array'

		if onscreen_render or mode == 'human':
			if self.viewer is None:
				self.viewer = mjp.MjViewerBasic(self.sim)
				#self.viewer.start()
			self.viewer.render()
			#self.viewer.loop_once()
			if mode == 'human':
				return

		rgb = self.sim.render(*img_dim,camera_name ='cam',depth=show_depth)
		if show_depth:
			rgb, depth = rgb
			depth = self.depth_to_linear(depth[::-1, :])  # transform to correct depth values
		rgb = rgb[::-1,:]
		#rgb = rgb[:,:,::-1]
		if not show_label:
			return (rgb, depth) if show_depth else rgb

		return (rgb, depth, self.label_maker.get_label(img_dim)) if show_depth else (rgb, self.label_maker.get_label(img_dim))


	def set_reset(self, state):
		''' reset to 'state' instead of randomly'''
		self.sim.data.qpos[0] = state[0]
		self.sim.data.qpos[1] = state[1]
		self.sim.data.qvel[0] = state[2]
		self.sim.data.qvel[1] = state[3]

		self.sim.forward()
		self.done = False
		return self._get_obs()

	def _get_obs(self): # [cart_pos, pole_pos, cart_vel, pole_vel]
		state = np.hstack([self.sim.data.qpos, self.sim.data.qvel])
		state[1] %= 2 * np.pi
		if state[1] >= np.pi:
			state[1] -= 2 * np.pi
		return state

	def reset(self):
		''' reset environment randomly - should be called before beginning a new episode'''
		self.sim.data.qpos[0] = np.random.uniform(low=self.observation_space.low[0]/4, high=self.observation_space.high[0]/4)
		if self.sigma >= np.pi:
			sample = np.random.uniform(low=self.observation_space.low[1], high=self.observation_space.high[1])
		else:
			sample = np.random.randn() * self.sigma
			if sample > self.observation_space.high[1]: sample = self.observation_space.high[1]
			if sample < self.observation_space.low[1]: sample = self.observation_space.low[1]
			self.sigma += self.increment
			if self.sigma > self.observation_space.high[1]: print('**Switching to uniform angle initialization')
		self.sim.data.qpos[1] = sample
		self.sim.data.qvel[0] = np.random.uniform(low=-0.1, high=0.1)
		self.sim.data.qvel[1] = np.random.uniform(low=-0.2, high=0.2)

		self.sim.forward()

		self.done = False
		return self._get_obs()
	
	def _dist_penalty(self, dist):
		return 10. / abs(abs(dist)-self.observation_space.high[0]) - 10. / self.observation_space.high[0]

	def step(self, action, n=1):
		''' take a step in the environment with the control 'action' '''

		if self.done:
			return self.last_ob, None, None, None
		
		raw_action = action

		if action > self.action_space.high[0]:
			action = self.action_space.high[0]
		if action < self.action_space.low[0]:
			action = self.action_space.low[0]
		self.sim.data.ctrl[0] = action
		for _ in range(n):
			self.sim.step()
			self.sim.data.ctrl[0] = 0.

		if self.pbc: # shift cart if it moves out of view
			if self.sim.data.qpos[0] < -self.keep_in_view:
				self.sim.data.qpos[0] += 2*self.keep_in_view
				self.sim.forward()
			elif self.sim.data.qpos[0] > self.keep_in_view:
				self.sim.data.qpos[0] -= 2*self.keep_in_view
				self.sim.forward()

		ob = self._get_obs()
		
		# penalize pendulum angle, giving a +1 reward if angle is smaller than pi/2 (0 is inverted)
		reward = 5 + 5 * np.cos(ob[1]) #+ int(np.abs(ob[1])<np.pi/2)
		
		# penalize large controls
		if np.abs(raw_action) > self.action_space.high[0]:
			reward -= 100 * (np.abs(raw_action) - self.action_space.high[0])
		
		# penalize moving the cart far away from the center
		dist_penalty = self._dist_penalty(ob[0])
		if dist_penalty > 10:
			dist_penalty = 10
		reward -= dist_penalty

		if isinstance(reward, np.ndarray):
			reward = reward[0]

		notdone = np.isfinite(ob).all()
		self.done = not notdone
		if self.keep_in_view and not (self.pbc or self.limited):
			self.done = self.done or abs(ob[0]) > self.keep_in_view
		if self.pbc:
			self.done = self.done or abs(self.sim.data.qvel[0]) > 20 # limit cart vel

		self.last_ob = ob
		
		return ob, reward, self.done, {}

	def evaluate_policy(self, agent_policy, num_episodes=10, remove_noise=False, horizon=None):
		horizon = horizon or self.horizon
		policy = agent_policy
		if remove_noise:
			policy = lambda o: agent_policy(o, add_noise=False)[0]

		agent_rewards = []
		agent_steps = []

		for ep in range(num_episodes):

			state = self.reset()
			done = False
			step = 0
			total_reward = 0

			while step < horizon and not done:
				step += 1
				state, reward, done, _ = self.step(policy(state))
				total_reward += reward

			agent_rewards.append(total_reward)
			agent_steps.append(step)

		return np.array(agent_rewards)

	def visualize_policy(self, agent_policy, num_episodes=10, remove_noise=False, horizon=None, onscreen=False):
		horizon = horizon or self.horizon
		policy = agent_policy
		if remove_noise:
			policy = lambda o: agent_policy(o, add_noise=False)[0]

		if not onscreen:
			fig, ax = plt.subplots()
			plt.ion()

		for ep in range(num_episodes):

			state = self.reset()
			done = False
			step = 0
			total_reward = 0
			action = [0.]

			while step < horizon and not done:
				step += 1
				rgb = self.render(onscreen_render=onscreen)
				if not onscreen:
					ax.set_title('Episode='+str(ep+1)+'/'+str(num_episodes)+'  step='+str(step)+'/'+str(horizon))
					#ax.set_ylabel('Action: ' + str(np.round(action[0],3)))
					ax.set_xlabel('Cumulative reward: '+ str(np.round(total_reward)) + '\n' + 'Action: ' + str(np.round(action[0],3)))
					ax.imshow(rgb)
					fig.tight_layout()
					#fig.draw()
					plt.pause(0.025)
					plt.cla()
				action = policy(state)
				state, reward, done, _ = self.step(action)
				total_reward += reward


class Img_Pendulum(Mujoco_Pendulum):
	'''
	Uses RGB and/or depth images as states
	'''

	def __init__(self, img_dim, depth=True, rgb=False, **kwargs):
		assert depth or rgb, 'State must consist of depth and/or rgb'
		self.img_dim = img_dim
		self.rgb, self.depth = rgb, depth
		super(Img_Pendulum, self).__init__(**kwargs)

	def reset(self):
		super(Img_Pendulum, self).reset()

		rgb, depth = self.render(self.img_dim)

		if self.rgb and self.depth:
			return (rgb, depth)
		if self.rgb:
			return rgb
		return depth


	def step(self, action):
		_, reward, done, info = super(Img_Pendulum, self).step(action)

		rgb, depth = self.render(self.img_dim)

		if self.rgb and self.depth:
			return (rgb, depth), reward, done, info
		if self.rgb:
			return rgb, reward, done, info
		return depth, reward, done, info





