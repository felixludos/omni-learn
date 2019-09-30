from mujoco_py import load_model_from_path, MjSim, MjViewer
import sys, os
import numpy as np

from foundation import util
from foundation import train
from foundation.envs.multi_object_env import Multi_Object
from foundation.envs.multi_object_table import Multi_Object_Table

num_obj = 3

env = Multi_Object_Table(num_obj=num_obj, xtrans=True, ytrans=True, ztrans=True, xrot=True, yrot=True, zrot=True,
				   bg_path='/home/fleeb/workspace/ml_datasets/places365/validation',
				   tex_path='/home/fleeb/workspace/ml_datasets/dtd')

print(type(env))
print(env.reset())
print(env.sim.model.body_names)


# plt.figure()
#
# for i in range(N):
# 	print(env.reset()[:3])
# 	for j in range(M):
# 		rgb, depth, lbl = env.render(160,120, show_depth=True, show_label=True)
# 		plt.cla()
# 		plt.imshow(rgb)
# 		plt.title('{}/{} : {}/{}'.format(i+1,N, j+1,M))
# 		plt.pause(0.01)
# 		action = np.random.rand(6)*2-1
# 		action *= 0
# 		env.step(action)




for i in range(5):

	env.reset()
	viewer = MjViewer(env.sim)

	for i in range(100):
		# if i < 170: # 150->red, 170->green
		# 	sim.data.ctrl[:] = 0.0
		# else:
		# 	sim.data.ctrl[:] = -1.0
		action = np.random.rand(env.action_dim) * 2 - 1
		action *= .00
		env.step(action)
		viewer.render()

		# print(action)

		# qpos = sim.data.qpos.copy()
		# qvel = sim.data.qpos.copy()
		#
		# print(i, qpos, qvel)



	#break

	if os.getenv('TESTING') is not None:
		break