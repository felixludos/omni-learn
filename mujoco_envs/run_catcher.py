from mujoco_py import load_model_from_path, MjSim, MjViewer
import sys, os
import numpy as np
import matplotlib.pyplot as plt


model = load_model_from_path("xmls/catcher.xml")
sim = MjSim(model)

view = True

if view:

	viewer = MjViewer(sim)


sim_state = sim.get_state()

print('state', sim_state)
print('ctrl', sim.data.ctrl)

base = 140 # good range 140 - 230

trigger_range = 140, 180
vr = 0.05*0
ar = 0.05*0
yr = 0.05
zr = 0.05

#bin_limits = np.array([.625, .75+.125, 1.125, 1.25+.125, 1.5+.125, 1.75+.125]) - 0.08
#print(bin_limits)

bins = []
triggers = []
vels = []
angles = []

N = 1000

print_freq = max(N//100, 1)

cnt = 0

maxes = []
xs = []
#_, edges = np.hist(xs, bins=20, range=(0,3))

#sim.model.body_mass[-1] = 1

print(sim.model.body_mass)

#plt.ion()
#plt.figure()
#for trigger in range(140, 230, 1):
for ep in range(N):

	trigger = np.random.randint(*trigger_range)
	#trigger = 150 - cnt #230 - cnt
	#trigger = 170 - cnt//2
	#print(trigger)



	vel = np.random.randn()*vr - vr/2
	angle = np.random.randn()*ar - ar/2

	triggers.append(trigger)
	vels.append(vel)
	angles.append(angle)

	sim.set_state(sim_state)
	sim.forward()

	#print(trigger)

	sim.data.qvel[3] = vel
	sim.data.qpos[4] = angle

	# print(sim.data.qpos)
	# print(sim.data.qvel)
	# quit()

	hit = False
	caught = False

	mx = None

	for i in range(1000):
		if i < trigger: # 150->red, 170->green
			sim.data.ctrl[1:] = 0.0
		else:
			sim.data.ctrl[1:] = -1.0
		# try:
		# 	sim.data.ctrl[:] = np.random.randn(len(sim.data.ctrl))
		# except:
		# 	pass
		#sim.data.ctrl[0] = -1 #np.random.rand()*2-1
		sim.step()
		if view:
			viewer.render()

		if i == 300:

			print(sim.data.qpos)
			print(sim.data.qvel)
			print(sim.data.sensordata)
			quit()

			img = sim.render(180,120, camera_name='cam', depth=False)

			plt.figure()
			plt.imshow(img)
			plt.show()

		if mx is None or sim.data.qpos[2] > mx:
			mx = sim.data.qpos[2]

		if not hit and sim.data.sensordata[-1] != 0:
			x = -sim.data.qpos[3]
			xs.append(x)
			hit = True
			#break

		# if not caught and sim.data.sensordata[-2] != 0:
		# 	print('caught', sim.data.sensordata)
		# 	caught = True

	maxes.append(mx)
	x = -sim.data.qpos[3]

	#b = np.searchsorted(bin_limits, x, side='right')
	#bins.append(b)

	if ep % print_freq == 0:
		#print('Ep {}/{}: {}'.format(ep+1, N, np.bincount(bins, minlength=8)))
		#print('Ep {}/{}: {}'.format(ep + 1, N, np.histogram(xs, bins=edges)[0]))
		print('Ep {}/{}'.format(ep+1, N))
		# plt.cla()
		# plt.title('Ep {}/{}'.format(ep+1, N))
		# plt.hist(xs, bins=20, range=(0,3))
		# plt.pause(0.0001)


	#print(sim.data.qpos)
	#print('Pos: {:.3f}, bin: {}'.format(x, )

	cnt += 1

plt.figure()
plt.cla()
plt.title('Ep {}/{}'.format(ep+1, N))
plt.hist(xs, bins=50, range=(0,3))

#plt.title('Ep {}/{}'.format(N, N))
plt.ioff()

#print(np.bincount(bins, minlength=8))

plt.figure()
plt.hist(maxes, bins=50)
plt.title('maxes')

# plt.figure()
# #plt.hist(triggers, bins=30)
# plt.bar(np.arange(*trigger_range), np.bincount(np.array(triggers)-trigger_range[0], minlength=8))
# plt.title('triggers')

# plt.figure()
# plt.hist(vels, bins=50)
# plt.title('vels')
#
# plt.figure()
# plt.hist(angles, bins=50)
# plt.title('angles')

plt.show()