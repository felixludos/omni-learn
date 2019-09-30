from mujoco_py import load_model_from_path, MjSim, MjViewer
import sys, os
import numpy as np


def sample_qpos(sim):
	x = np.random.randn(*sim.data.qpos.shape)

	mlim = sim.model.jnt_limited == 1
	if mlim.sum() > 0:
		x[sim.model.jnt_qposadr[mlim]] = np.random.uniform(*sim.model.jnt_range[mlim].T)

	return x


try:
	path = sys.argv[1]
	assert path[-4:] == '.xml'
except:
	path = "xmls/tosser.xml"

model = load_model_from_path(path)
sim = MjSim(model)

viewer = MjViewer(sim)


sim_state = sim.get_state()

print('state', sim_state)
print('ctrl', sim.data.ctrl)



while True:

	qpos = sample_qpos(sim)
	sim_state.qpos[:] = qpos
	sim.set_state(sim_state)

	#img = sim.render(320, 240)
	#print(img.shape)
	#quit()

	#sim.set_state(sim_state)
	#sim.data.qpos[:] = sample_state()

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

	if os.getenv('TESTING') is not None:
		break