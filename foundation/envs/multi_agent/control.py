import numpy as np
import torch
from .. import general as gen
from ... import framework as fm

'''
Planned envs:

- Mass Balance: agents must match a goal number over a sequence of actions by taking 1 of 4 actions each timestep. The noise and mass coeffs of each agent are fixed by different:
	- Observe: get a noisy estimate of the difference between agents' current estimate and goal
	- Add: add to the estimate
	- Remove: remove from the estimate
	- Communicate:

- Electric Trap: each agent controls the charge of a different point charge or set of charges (maybe with overlap).
        Env contains a moving charge, and the agents must keep it inside the frame.

- Walking: split joints of a mujoco env (eg. ant) and give each agent a different set of joints (eg. one appendage).
	- One-Way: commander receives all joints and gives agents discrete orders, agents only observe their joints
	- Two-Way: commander only observes discrete comms from agents.
'''

class Mass_Balance(fm.Env):
	def __init__(self, num_agents=4, com_channels=2, time_factor=3, seq_len=20, inv_skill=1):
		raise Exception('not working')
		self.obs_shape = num_agents, com_channels*num_agents + 1
		self.A = num_agents
		
		self.C = com_channels
		self.horizon = seq_len
		self.spec = gen.EnvSpec(obs_space=gen.Continuous_Space(shape=self.obs_shape,),
		                        act_space=gen.Discrete_Space(choices=com_channels + 3, shape=(num_agents,)),
		                        horizon=seq_len,
		                        num_agents=num_agents,
		                        obs_masks=torch.eye(self.A, dtype=torch.uint8).unsqueeze(-1).expand(self.A, self.A, self.obs_shape[-1]),#.numpy().astype(bool),
		                        act_masks=torch.eye(self.A, dtype=torch.uint8))#np.eye(self.A, dtype=bool))
		
		super(Mass_Balance, self).__init__(self.spec)
		
		# these params stay constant for all episodes
		self.weights = torch.rand(self.A)
		self.obs_noise = self.weights * inv_skill if inv_skill > 0 else torch.zeros(self.A)
		self.goal_mag = self.weights.max() * seq_len / time_factor # mean goal mag
		self.goal_std = self.weights.max()
		
	def _measurements(self):
		return self.obs_noise * torch.randn(self.A) + (self.goal - self.estimate)
		
	def reset(self, init_state=None):
		self.goal = (-1)**np.random.randint(2) * (self.goal_mag + np.random.randn() * self.goal_std)
		self.estimate = 0.
		self.current_step = 0
		
		blank = torch.zeros(*self.obs_shape)
		blank[:,-1] = self._measurements()
		return blank
	
	def step(self, actions):
		
		measurements = self._measurements()
		full_obs = torch.zeros(*self.obs_shape)
		
		for i, (a, obs, m, w) in enumerate(zip(actions, full_obs, measurements, self.weights)):
			if a < self.C:
				full_obs[:,i] = 1
			elif a == self.C: # observe
				obs[-1] = m
			elif a == self.C+1: # add
				self.estimate += w
			elif a == self.C+2: # remove
				self.estimate -= w
			
		reward, done = 0, False
		if self.current_step == self.horizon:
			reward = np.abs(self.goal - self.estimate)
			done = True
		self.current_step += 1
		
		return full_obs, reward, done, {}

class Walking(gen.Multi_Agent_Env):
	def __init__(self, commander=True, cmd2sub_channels=3, sub2cmd_channels=3, cut_comm_graph=True):

		self._ant = gen.GymEnv('SimpleAnt-v0')
		self.cut_comm_graph = cut_comm_graph # cut graph of comms
		self.training = True
		self.commander = True

		assert commander
		assert self.cut_comm_graph
		self.c2s_C = cmd2sub_channels
		self.s2c_C = sub2cmd_channels

		self._horizon = self._ant.spec.horizon
		

		cmd_spec = gen.EnvSpec(
			obs_space=gen.Continuous_Space(shape=(4*self.s2c_C+12,), high=1, low=0),
			act_space=gen.Continuous_Space(shape=(4*self.c2s_C,), high=1, low=0),
			horizon=self.horizon,
		)
		sub_spec = gen.EnvSpec(
			obs_space=gen.Continuous_Space(shape=(4+self.c2s_C,)),
			act_space=gen.Continuous_Space(shape=(2+self.s2c_C,)),
			horizon=self.horizon,
		)

		super(Walking, self).__init__([cmd_spec]+4*[sub_spec], ID='WalkingAnt-v0')

	def _get_obs(self, joints): # not including comms
		cmd_obs = torch.zeros(self.spec[0].obs_space.size)
		cmd_obs[-12:] = joints[-12:]
		sub_obs = [torch.zeros(self.spec[1].obs_space.size) for _ in range(4)]
		for i, sub in enumerate(sub_obs):
			p = i * 2
			sub[:2] = joints[p:p + 2]
			v = p + 8
			sub[2:4] = joints[v:v + 2]

		return [cmd_obs] + sub_obs

	def reset(self, init_state=None):
		joints = torch.from_numpy(self._ant.reset(init_state)) # (28,)

		return self._get_obs(joints)

	def step(self, actions):
		#assert len(actions) == 5

		jnt_cmds = torch.cat([a[:2] for a in actions[1:]]).detach().cpu().numpy()

		joints, reward, done, info = self._ant.step(jnt_cmds)

		next_obs = self._get_obs(torch.from_numpy(joints))

		if self.cut_comm_graph: # threshold comms
			for a in actions:
				a = a.detach()
				#a[a>=0.5] = 1
				#a[a<0.5] = 0

		for i, (sub,com) in enumerate(zip(next_obs[1:], actions[1:])):

			# cmd comms
			s = i*self.s2c_C
			next_obs[0][s : s+self.s2c_C] = com[2:]

			# sub comms
			c = i*self.c2s_C
			sub[4:] = actions[0][c : c+self.c2s_C]

		return next_obs, [reward]*5, done, info

	def evaluate_policy(self, N, policy, T=None, init_state=None, seed=None, render=False):
		#print('walking eval {}'.format(render))
		self.training = False
		self._ant.env.env.mujoco_render_frames = render
		ret = super(Walking, self).evaluate_policy(N, policy, T=T, init_state=init_state, seed=seed, render=False)
		self._ant.env.env.mujoco_render_frames = False
		self.training = True
		return ret