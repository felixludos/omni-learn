
import sys, os
import numpy as np
import mujoco_py as mjp
from collections import namedtuple
from itertools import product

MY_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Mujoco_PR2_Batting(object):
	'''
	Torque control on a PR2 arm holding a bat.
	A ball is moving towards the robot's head, which must be swatted away using the bat (no collisions between robot arm and ball)

	State space: (20 dim) [7 joint angles] + [3 ball pos (xyz)] + [7 joint vel] + [3 ball vel]
	Action space: [7 joint efforts (torques) (min=-1,max=1)]

	Two reward settings:
		sparse: -1 if ball hits robot's head, 0 otherwise
		dense: distance between robot's head and ball

	Max episode length: 1000 steps (at 200 Hz)
	Episode automatically ends if ball collides with robot head or if ball is moving away from the head

	'''

	def __init__(self, model_file=None, reward_type='sparse'):
		self.env_id = 'MjpPR2Batting-v1'
		self.model_file = model_file if model_file is not None else os.path.join(MY_DIR_PATH,
		                                                                         'xmls/pr2_batting.xml')
		self.model = mjp.load_model_from_path(self.model_file)

		self.sim = mjp.MjSim(self.model)
		assert self.sim.model.stat.extent == 1, 'set the extent to 1 in the xml'

		mlim = self.sim.model.jnt_limited == 1
		min, max = self.sim.model.jnt_range[mlim].T

		self.neutral_pos = np.zeros(7)
		self.neutral_pos[self.sim.model.jnt_qposadr[mlim]] = (max + min) / 2
		self.neutral_pos[4] = 0
		self.neutral_pos[6] = -np.pi / 2

		self.head_pos = np.array([-0.15, 0., 1]) # approximate center of head for dense rewards

		self.head_idx = self.sim.model.geom_name2id('geom15')
		self.ball_idx = self.sim.model.geom_name2id('ball')
		self.bat_idx = self.sim.model.geom_name2id('bat')

		self.viewer = None  # mjp.MjViewerBasic(self.sim)

		assert reward_type in {'sparse', 'dense'}, 'unknown reward type {}'.format(reward_type)
		self.reward_type = reward_type

		self.obs_dim = (7+3)*2
		self.action_dim = 7
		self.horizon = 1000

	def render(self, mode='human', img_ht=240, img_wd=320):

		if mode == 'human':
			if self.viewer is None:
				self.viewer = mjp.MjViewer(self.sim)
			self.viewer.render()
			return

		raise NotImplementedError # RGB rendering not supported yet
		return self.sim.render(*img_dim, camera_name='cam')

	def _get_obs(self):
		state = np.concatenate([self.sim.data.qpos[:10], self.sim.data.qvel[:10]])
		return state

	def _check_collisions(self):

		hit, miss = False, False

		for col in self.sim.contact:
			g1, g2 = col.geom1, col.geom2
			if g1 == g2 == 0:
				break
			if (self.ball_idx==2 and g2==self.head_idx) or (g1==self.head_idx and self.ball_idx==2):
				miss = True
			if (g1==self.ball_idx and g2==self.bat_idx) or (g1==self.bat_idx and g2==self.ball_idx):
				hit = True
		return hit, miss


	def reset(self, state=None):
		''' reset environment randomly - should be called before beginning a new episode'''
		if state is None:

			state = np.zeros(20)

			# arm always starts in neutral pos
			state[:7] = self.neutral_pos

			# select init pos for ball
			init = np.random.uniform([1.4, -1.5, 0], [1.6, 1.5, 2])
			state[7:10] = init

			# select target for ball
			target = np.random.uniform([-.1, -.2, 1], [.1, .2, 1.1])
			diff = target - init
			dir = diff / np.sqrt((diff ** 2).sum())
			# select speed for ball
			speed = np.random.uniform(0.5, 1.5)

			state[17:20] = speed * dir

		sim_state = self.sim.get_state()

		sim_state.qpos[:10] = state[:10]
		sim_state.qvel[:10] = state[10:]

		self.sim.set_state(sim_state)
		self.sim.forward()

		self.is_miss = False # ball hit robot
		self.is_hit = False # bat hit ball

		ob = self._get_obs()

		self.init_vel = ob[17:20].copy()

		return ob

	def step(self, action, n=10):
		''' take a step in the environment with the control 'action' '''

		action = action.clip(-1,1)

		self.sim.data.ctrl[:] = action
		for _ in range(n):
			self.sim.step()

		ob = self._get_obs()

		hit, miss = self._check_collisions()

		self.is_miss = self.is_miss or miss # True as long as ball is hitting head
		self.is_hit = hit and not self.is_hit # only True for 1 timestep when first collided

		done = miss or (self.init_vel @ ob[17:20] < 0)

		reward = 0.
		if self.reward_type == 'sparse' and miss:
			reward = -1.
		elif self.reward_type == 'dense':
			ball_pos = ob[7:10]
			dist = np.sqrt(((ball_pos - self.head_pos)**2).sum())
			reward = dist

		return ob, reward, done, {}




