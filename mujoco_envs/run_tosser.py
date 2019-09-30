from mujoco_py import load_model_from_path, MjSim, MjViewer
import sys, os
import numpy as np
import matplotlib.pyplot as plt


model = load_model_from_path("xmls/tosser.xml")
sim = MjSim(model)

view = True

if view:

	viewer = MjViewer(sim)


sim_state = sim.get_state()

print('state', sim_state)
print('ctrl', sim.data.ctrl)

base = 140 # good range 140 - 230

bin_limits = np.array([.5-.125, .625, .75+.125, 1.125, 1.25+.125, 1.5+.125, 1.75+.125]) - 0.08

#bin_limits = np.array([.625, .75+.125, 1.125, 1.25+.125, 1.5+.125, 1.75+.125]) - 0.08
#print(bin_limits)

bins = []
triggers = []
vels = []
angles = []

N = 1000

print_freq = max(N//100, 1)

cnt = 0

#for trigger in range(140, 230, 1):
for ep in range(N):

	trigger = np.random.randint(140, 240)
	vel = np.random.randn()*0.1
	angle = np.random.randn()*.1

	triggers.append(trigger)
	vels.append(vel)
	angles.append(angle)

	sim.set_state(sim_state)

	#print(trigger)

	sim.data.qvel[4] = vel
	sim.data.qpos[-1] = angle

	for i in range(1000):
		if i < trigger: # 150->red, 170->green
			sim.data.ctrl[:] = 0.0
		else:
			sim.data.ctrl[:] = -1.0
		# try:
		# 	sim.data.ctrl[:] = np.random.randn(len(sim.data.ctrl))
		# except:
		# 	pass
		sim.step()
		if view:
			viewer.render()

	x = -sim.data.qpos[3]

	b = np.searchsorted(bin_limits, x, side='right')
	bins.append(b)

	if ep % print_freq == 0:
		print('Ep {}/{}: {}'.format(ep+1, N, np.bincount(bins, minlength=8)))

	#print(sim.data.qpos)
	#print('Pos: {:.3f}, bin: {}'.format(x, )

	cnt += 1

print(np.bincount(bins, minlength=8))

plt.figure()
plt.bar(range(8), np.bincount(bins, minlength=8))
plt.title('bins')

plt.figure()
plt.hist(triggers, bins=50)
plt.title('triggers')

plt.figure()
plt.hist(vels, bins=50)
plt.title('vels')

plt.figure()
plt.hist(angles, bins=50)
plt.title('angles')

plt.show()