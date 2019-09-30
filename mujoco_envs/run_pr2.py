from mujoco_py import load_model_from_path, MjSim, MjViewer
import sys, os
import numpy as np


def sample_qpos(sim):
	x = np.random.randn(*sim.data.qpos.shape)

	mlim = sim.model.jnt_limited == 1
	if mlim.sum() > 0:
		x[sim.model.jnt_qposadr[mlim]] = np.random.uniform(*sim.model.jnt_range[mlim].T)

	x[-7:] = [.5,0,1,1,0,0,0]

	return x

def set_qpos(sim, sim_state):
	x = np.random.randn(*sim.data.qpos.shape)*0

	mlim = sim.model.jnt_limited == 1
	min, max = sim.model.jnt_range[mlim].T

	x[sim.model.jnt_qposadr[mlim]] = (max + min) / 2
	x[4] = 0
	x[6] = -np.pi/2


	x[-7:-4] = [0, -.2, .9] # target [0, -.2 to .2, .9 to 1.1]

	x[-7:-4] = np.random.uniform([1.4, -1.5, 0], [1.6, 1.5, 2]) # init [1.5, -1.5 to 1.5, 0 to 2]

	x[-4:] = [1,0,0,0]



	#print(x) # [ 0.785398    0.43635     1.55       -1.16065     0.         -1.047
			 # -1.57079633  1.          0.          1.          1.          0.
			 #  0.          0.        ]

	#return x
	sim_state.qpos[:] = x

def set_qvel(sim, sim_state):
	target = np.random.uniform([-.1, -.2, 1], [.1, .2, 1.1])

	pos = sim_state.qpos[-7:-4]

	diff = target-pos
	dir = diff / np.sqrt((diff**2).sum())

	speed = np.random.uniform(0.5, 1.2)

	vel = np.zeros(len(sim_state.qvel))

	vel[-6:-3] = speed * dir

	sim_state.qvel[:] = vel


path = 'xmls/pr2_batting.xml'

model = load_model_from_path(path)
sim = MjSim(model)

viewer = MjViewer(sim)


sim_state = sim.get_state()

print('state', sim_state)
print('ctrl', sim.data.ctrl)

import time

while True:

	set_qpos(sim, sim_state)
	set_qvel(sim, sim_state)

	sim.set_state(sim_state)

	#sim.set_state(sim_state)
	#sim.data.qpos[:] = sample_state()

	tick = time.time()

	for step in range(1000):
		# if i < 170: # 150->red, 170->green
		# 	sim.data.ctrl[:] = 0.0
		# else:
		# 	sim.data.ctrl[:] = -1.0
		try:
			sim.data.ctrl[:] = np.random.randn(len(sim.data.ctrl))
		except:
			pass
		sim.step()

		g1s = [contact.geom1 for contact in sim.data.contact]
		g2s = [contact.geom2 for contact in sim.data.contact]

		print(sim.data.geom_xpos[sim.model.geom_name2id('target')])
		quit()

		colls = list(zip(g1s, g2s))
		#print(colls)
		is_done = False
		for i,j in colls:
			if (i==2 and j==28) or (i==28 and j==2):
				#print('\n'*4)
				#print(colls)
				print('done')
				is_done = True
				break

			if (i==12 and j==28) or (i==28 and j==12):
				#print('\n'*4)
				#print(colls)
				print('hit')

		if is_done:
			break

		#print()

		viewer.render()

	tock = time.time()
	print('{} steps -  {:.3f} s/sec'.format(step+1, (step+1)/(tock-tick)))

	if os.getenv('TESTING') is not None:
		break