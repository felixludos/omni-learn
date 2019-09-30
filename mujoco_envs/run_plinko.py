from mujoco_py import load_model_from_path, MjSim, MjViewer
import sys, os
import numpy as np


path = 'xmls/plinko.xml'

model = load_model_from_path(path)
sim = MjSim(model)

viewer = MjViewer(sim)


sim_state = sim.get_state()

print('state', sim_state)
print('ctrl', sim.data.ctrl)



while True:
	sim.set_state(sim_state)

	sim.data.qpos[0] = .93 * np.random.rand()

	for i in range(1000):
		# if i < 170: # 150->red, 170->green
		# 	sim.data.ctrl[:] = 0.0
		# else:
		# 	sim.data.ctrl[:] = -1.0
		try:
			sim.data.ctrl[:] = np.random.randn(len(sim.data.ctrl))
		except:
			pass
		sim.step()
		viewer.render()

		# qpos = sim.data.qpos.copy()
		# qvel = sim.data.qpos.copy()
		#
		# print(i, qpos, qvel)

	#break

	if os.getenv('TESTING') is not None:
		break