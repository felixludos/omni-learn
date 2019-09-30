from mujoco_py import load_model_from_xml, MjSim, MjViewer
import sys, os, time
import numpy as np
import foundation as fd
from foundation import util

#from domain_randomization_mujoco import *
from foundation.sim.mujoco import *

def sample_qpos(sim):
	x = np.random.rand(*sim.data.qpos.shape)*2*np.pi

	# print(sim.model.jnt_range)
	# quit()

	mlim = sim.model.jnt_limited == 1
	if mlim.sum() > 0:
		x[sim.model.jnt_qposadr[mlim]] = np.random.uniform(*sim.model.jnt_range[mlim].T)

	#print(x)

	return x

tex_path = '/home/fleeb/Downloads/dtd/images/'
bg_path = '/home/fleeb/Downloads/val_256/'
texture_files = util.crawl(tex_path, lambda x: '.jpg' in x)
bg_files = util.crawl(bg_path, lambda x: '.png' in x)

tmpl_path = '../foundation/envs/assets/multi_object_template.xml'
with open(tmpl_path, 'r') as f:
	model_tmpl = f.read()
num_prim = 3

geoms, materials, textures = gen_object(num_prim)
names = ['geom', 'material', 'texture']

#geoms, materials, textures = [[xml_format(name, a) for a in attr] for attr, name in zip(attrs, names)]
textures = textures + [gen_skybox(bg_files)]
materials = materials + gen_label_mats()

geoms = [xml_format('geom', g) for g in geoms]
textures = [xml_format('texture', g) for g in textures]
materials = [xml_format('material', g) for g in materials]

assets = materials + textures

model_str = model_tmpl.format('\n\t\t\t'.join(assets), '\n\t\t\t'.join(geoms))


model = load_model_from_xml(model_str)
sim = MjSim(model)
#viewer = MjViewer(sim)

lbl_maker = Label_Maker(sim, body_names=['floating_obj'])

sim_state = sim.get_state()

N = 50

render_viewer = True
try:
	viewer = MjViewer(sim)
except:
	render_viewer = False
	print('using matplotlib')

if render_viewer:
	import glfw
else:
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()

def depth_to_linear(depth):  # zNear and zFar values must set to 0.2 and 3.0 and model extent must be set to 1 in model xml
	zNear, zFar = sim.model.vis.map.znear, sim.model.vis.map.zfar
	return zFar * zNear / (zFar + depth * (zNear - zFar))

while True:
	#print(sim.model.geom_type)
	#print(sim._render_context_offscreen)
	#quit()
	#sim.reset()
	sim = MjSim(model)
	#sim.model.geom_type[0] = 2
	#sim.model.geom_size[0,0] = 0.1
	#print(sim.model.geom_type)
	sim.reset()

	#print(sim.model.mat_rgba)
	#print(sim.model.mat_emission)

	tex = pick_textures(sim, texture_files, bg_files, bg_idx=num_prim)
	texture_geoms(sim, tex)

	#sim.reset()

	randomize_obj(sim, 'floating_obj')

	#sim.model.geom_matid[:] = -1

	#print(sim.model.geom_quat)
	#print(np.linalg.norm(sim.model.geom_quat,axis=-1))

	#set_simple_texture(sim, 3)
	if render_viewer:
		viewer = MjViewer(sim)

	qpos = sample_qpos(sim)
	sim_state.qpos[:] = 0#qpos
	sim_state.qvel[:] = np.random.randn(len(sim_state.qvel))
	sim.set_state(sim_state)

	#img = sim.render(320, 240)
	#print(img.shape)
	#quit()

	#sim.set_state(sim_state)
	#sim.data.qpos[:] = sample_state()

	# idx = 1
	#
	# s = sim.model.tex_adr[idx]
	# e = sim.model.tex_adr[idx + 1] if (idx + 1) < len(sim.model.tex_adr) else len(sim.model.tex_rgb)
	# h, w = sim.model.tex_height[idx], sim.model.tex_width[idx]
	# data = sim.model.tex_rgb[s:e].reshape(h, w, 3)
	# plt.figure()
	# plt.imshow(data)
	#plt.axis('off')



	tick = time.time()

	for i in range(N):
	#while True:
		# if i < 170: # 150->red, 170->green
		# 	sim.data.ctrl[:] = 0.0
		# else:
		# 	sim.data.ctrl[:] = -1.0
		try:
			a = np.random.randn(len(sim.data.ctrl))
			#a[-1] = 1
			sim.data.ctrl[:] = a
		except:
			pass
		sim.step()
		if render_viewer:
			viewer.render()
		else:
			#print(sim.model.geom_size)
			rgb, depth = sim.render(320, 240, camera_name='external_camera_0', depth=True)
			lbl = lbl_maker.get_label(320, 240, sim)

			img = rgb #lbl

			plt.sca(ax)
			plt.cla()
			plt.imshow(img)
			#plt.imshow(depth_to_linear(depth))
			plt.pause(0.001)

	tock = time.time()

	if render_viewer:
		glfw.destroy_window(viewer.window)
		del viewer

	print('{:.3f}'.format(N/(tock-tick)))
