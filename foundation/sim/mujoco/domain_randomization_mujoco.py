
import sys, os, time
import numpy as np
import mujoco_py as mjp
import cv2

from ... import util
# from ...util import get_patch, get_img

import torch
from torch import nn

'''<body name="floating_obj" pos="0 0 0">

            <inertial pos="0 0 0" mass="1" diaginertia=".25 .25 .25"/>


            <joint name="obj_jnt_x" pos="0 0 0" axis="1 0 0" type="slide" limited='true' range="-.5 .5" damping="0.01" />
            <joint name="obj_jnt_y" pos="0 0 0" axis="0 1 0" type="slide" limited='true' range="-.5 .5" damping="0.01" />
            <joint name="obj_jnt_z" pos="0 0 0" axis="0 0 1" type="slide" limited='true' range="-.5 .5" damping="0.01" />

            <joint name="obj_jnt_wx" pos="0 0 0" axis="1 0 0" type="hinge" limited='false' damping="0.01" />
            <joint name="obj_jnt_wy" pos="0 0 0" axis="0 1 0" type="hinge" limited='false' damping="0.01" />
            <joint name="obj_jnt_wz" pos="0 0 0" axis="0 0 1" type="hinge" limited='false' damping="0.01" />

            {}
        </body>'''

def get_jnt_strs(idx, trans_gear = 16, rot_gear = 4, damping=0.01, limited=None, pos_lim=0.5,
				 pos_ctrl=None, no_jnts=False):

	if limited is None:
		limited = [pos_lim, pos_lim, pos_lim, 0, 0, 0]

	jnt_strs = [
		'<joint name="obj{}_jnt_x" pos="0 0 0" axis="1 0 0" type="slide" limited="{}" range="-{} {}" damping="{}" />',
		'<joint name="obj{}_jnt_y" pos="0 0 0" axis="0 1 0" type="slide" limited="{}" range="-{} {}" damping="{}" />',
		'<joint name="obj{}_jnt_z" pos="0 0 0" axis="0 0 1" type="slide" limited="{}" range="-{} {}" damping="{}" />',

		'<joint name="obj{}_jnt_wx" pos="0 0 0" axis="1 0 0" type="hinge" limited="{}" damping="{}" />',
		'<joint name="obj{}_jnt_wy" pos="0 0 0" axis="0 1 0" type="hinge" limited="{}" damping="{}" />',
		'<joint name="obj{}_jnt_wz" pos="0 0 0" axis="0 0 1" type="hinge" limited="{}" damping="{}" />', ]

	jnt_strs = [
		'<joint name="obj{}_jnt_x" pos="0 0 0" axis="1 0 0" type="slide" limited="{}" range="-{} {}" damping="{}" />',
		'<joint name="obj{}_jnt_y" pos="0 0 0" axis="0 1 0" type="slide" limited="{}" range="-{} {}" damping="{}" />',
		'<joint name="obj{}_jnt_z" pos="0 0 0" axis="0 0 1" type="slide" limited="{}" range="-{} {}" damping="{}" />',

		'<joint name="obj{}_jnt_rot" pos="0 0 0" type="ball" limited="{}" damping="{}" />',

	]
	jnt_strs = [s.format(idx, 'true' if lim>0 else 'false', lim, lim, damping) for s, lim in zip(jnt_strs, limited)]

	act_trans_strs = ['<motor gear="{}" joint="obj{}_jnt_x" name="obj{}:motionX"/>',
					  '<motor gear="{}" joint="obj{}_jnt_y" name="obj{}:motionY"/>',
					  '<motor gear="{}" joint="obj{}_jnt_z" name="obj{}:motionZ"/>', ]
	act_rot_strs = ['<motor gear="{}" joint="obj{}_jnt_wx" name="obj{}:motionWX"/>',
					'<motor gear="{}" joint="obj{}_jnt_wy" name="obj{}:motionWY"/>',
					'<motor gear="{}" joint="obj{}_jnt_wz" name="obj{}:motionWZ"/>', ]
	act_rot_strs = [
		'<motor gear="{} 0 0" joint="obj{}_jnt_rot" name="obj{}:motionWX"/>',
		'<motor gear="0 {} 0" joint="obj{}_jnt_rot" name="obj{}:motionWY"/>',
		'<motor gear="0 0 {}" joint="obj{}_jnt_rot" name="obj{}:motionWZ"/>',
	]
	if pos_ctrl is not None:
		act_rot_strs = ['<position kp="{}" joint="obj{}_jnt_wx" name="obj{}:motionWX"/>',
						'<position kp="{}" joint="obj{}_jnt_wy" name="obj{}:motionWY"/>',
						'<position kp="{}" joint="obj{}_jnt_wz" name="obj{}:motionWZ"/>', ]
		rot_gear = 10

	act_strs = [s.format(trans_gear, idx, idx) for s in act_trans_strs] + [s.format(rot_gear, idx, idx) for s in
																		  act_rot_strs]

	if no_jnts:
		jnt_strs = ['' for j in jnt_strs]
		act_strs = ['' for a in act_strs]

	return jnt_strs, act_strs

obj_idx = 0
def create_free_objs(num=1, num_prim=3, pos_ctrl=None,
						no_jnts=False,
					 lbl_mats=True, pos_lim=0.5, damping=0.01, mass=1,
                     use_floor=False, limit_z=.5): # returns full model string
	# model file should be missing: assets, bodies, joints
	# model file should NOT be missing: camera/light

	inertial = '<inertial pos="0 0 0" mass="1" diaginertia=".25 .25 .25"/>'

	bodies = []
	acts = []
	texs = []
	mats = []

	texs.append(gen_skybox(use_floor)) # bg_files
	if lbl_mats:
		mats.extend(gen_label_mats())

	if use_floor:
		floor_tex, floor_mat, floor_body = gen_floor()
		texs.append(floor_tex)
		mats.append(floor_mat)
		bodies.append(floor_body)

	limited = [pos_lim, pos_lim, pos_lim, 0, 0, 0]
	limited[2] = limit_z

	for idx in range(num):

		head = '<body name="obj{}" pos="0 0 0">'.format(idx)
		tail = '</body>'

		jnts, act = get_jnt_strs(idx, limited=limited, pos_lim=pos_lim, damping=damping,
								 no_jnts=no_jnts, pos_ctrl=pos_ctrl)

		geoms, mat, tex = gen_object_geoms(idx, num_prim, mass=mass)
		geoms = '\n\t'.join([xml_format('geom', g) for g in geoms])

		obj = '''{head}
	{inertial}
	{jnts}
	{geoms}
{tail}'''.format(head=head, tail=tail, geoms=geoms, inertial=inertial, jnts = '\n\t'.join(jnts))

		bodies.append(obj)
		acts.extend(act)
		texs.extend(tex)
		mats.extend(mat)


	assets = [xml_format('material', m) for m in mats] + [xml_format('texture', t) for t in texs]

	return '\n'.join(bodies), '\n'.join(acts), '\n'.join(assets)

def gen_floor():

	mat = {
		'name':'floor_mat',
		'emission': '100',
		'shininess': '0',
		'specular': '0',
		'reflectance': '0',

		'texture': 'floor_tex',
	}

	tex = {
		'name': 'floor_tex',
		'type': '2d',

		'file': '/home/fleeb/workspace/marl/foundation/mujoco_envs/noise.png',
	}

	body = ' <geom  material="floor_mat" name="floor" pos="0 0 -5" size="2 2 2" type="plane"/>'

	body = '<geom  material="floor_mat" name="floor" pos="0 0 -0.1" size="1.1 1.1 .1" type="box"/>'

	return tex, mat, body


def depth_to_linear(depth, zNear, zFar):
	return zFar*zNear / (zFar + depth * (zNear - zFar))

def xml_format(name, attrs):
	attrs = [k + '="' + str(v) + '"' for k, v in attrs.items()]
	return '<{} {}/>'.format(name, ' '.join(attrs))

def gen_label_mats():
	mats = []

	mat = {}
	mat['name'] = 'label_material_k'
	rgb = [0, 0, 0]
	mat['rgba'] = '{} {} {} 1'.format(*rgb)
	mat['specular'] = '0'
	mat['shininess'] = '0'
	mat['reflectance'] = '0'
	mat['emission'] = '1000'
	mats.append(mat)

	for i, c in enumerate('rgb'):
		mat = {}
		mat['name'] = 'label_material_{}'.format(c)
		rgb = [0,0,0]
		rgb[i] = 1
		mat['rgba'] = '{} {} {} 1'.format(*rgb)
		mat['specular'] = '0'
		mat['shininess'] = '0'
		mat['reflectance'] = '0'
		mat['emission'] = '1000'
		mats.append(mat)
	return mats

class Label_Maker(object):
	def __init__(self, sim, body_names=None, cam_name='external_camera_0'):
		self.sim = sim
		self.cam_name = cam_name
		if body_names is None:
			body_names = {name for name in self.sim.model.body_names if name != 'world' and 'camera' not in name}
			print('WARNING: automatically choosing bodies to label: {}'.format(body_names))
		else:
			for name in body_names:
				assert name in self.sim.model.body_names, 'Error: couldnt find {} in body names'.format(name)
		self.body_names = list(body_names)  # ordered

		self.original_mats = sim.model.geom_matid.copy()
		# self.sels = [sim.model.body_name2id(body)==sim.model.geom_bodyid for body in self.body_names]
		self.lbl_mids = np.arange(len(sim.model.mat_emission))[np.isclose(sim.model.mat_emission, 1000)]
		# assert len(self.lbl_mids) >= len(self.sels), 'not enough label materials'
		self.lbl_black, *self.lbl_mids = self.lbl_mids

		self.lblmats = []
		for i in range((len(body_names)+len(self.lbl_mids)-1) // len(self.lbl_mids)):

			mats = np.ones(len(sim.model.geom_matid))*self.lbl_black

			for mid, body in zip(self.lbl_mids, self.body_names[i*len(self.lbl_mids):(i+1)*len(self.lbl_mids)]):
				mats[sim.model.body_name2id(body)==sim.model.geom_bodyid] = mid

			self.lblmats.append(mats)

		if len(self.lblmats) > 1:
			print('WARNING: Each label requires {} rendering steps'.format(len(self.lblmats)))

		self.screen_idx = self.sim.model.geom_name2id('screen')

	def get_key(self):
		return {name: i + 1 for i, name in enumerate(self.body_names)}

	def get_label(self, wd, ht, sim=None):
		if sim is not None:
			self.sim = sim

		lbl = None

		# self.sim.model.geom_matid[self.screen_idx] = self.lbl_black

		for i, mats in enumerate(self.lblmats):

			# label
			self.sim.model.geom_matid[:] = mats

			# render
			raw_lbl = self.sim.render(wd, ht, camera_name=self.cam_name)
			raw_lbl = raw_lbl[::-1, :]


			# unlabel
			self.sim.model.geom_matid[:] = self.original_mats

			# if i == len(self.lblmats)-1:
			# return raw_lbl

			bg_sel = np.sum(raw_lbl, axis=2) == 0

			newlbl = np.argmax(raw_lbl, axis=2) + 1 + i*len(self.lbl_mids)
			newlbl[bg_sel] = 0

			if lbl is None:
				lbl = newlbl
			else:
				sel = lbl==0
				lbl[sel] = newlbl[sel]

		# self.sim.model.mat_rgba[self.screen_idx, -1] = 0

		return lbl


_types = ['sphere', 'capsule', 'ellipsoid', 'cylinder', 'box']

def gen_object_geoms(oid=0, num=3, mass=1):
	idx = np.random.randint(0, 5, num)
	geoms = []
	textures = []
	materials = []
	masses = np.random.rand(num)
	masses /= masses.sum() # total mass is 1
	masses *= mass

	pos = np.random.randn(num, 3)
	pos *= 0.1 / np.sqrt((pos**2).sum(-1,keepdims=True))
	pos -= pos.mean(0, keepdims=True)
	#pos *= 0

	for n, (i, m, p) in enumerate(zip(idx, masses, pos)):
		geom = {}
		geom['type'] = 'sphere' #_types[i]
		sizes = np.random.rand(3) * 0.1*0 + 0.1
		euler = np.random.rand(3) * np.pi * 2
		geom['size'] = ' '.join(['{:.4f}'.format(s) for s in sizes])
		geom['name'] = 'geom{}-{}'.format(oid,n)
		geom['pos'] = ' '.join(['{:.4f}'.format(s) for s in p])
		geom['euler'] = ' '.join(['{:.4f}'.format(s) for s in euler])
		geom['material'] = 'mat{}-{}'.format(oid,n)
		geom['mass'] = m
		geoms.append(geom)

		mat = {}
		mat['name'] = 'mat{}-{}'.format(oid,n)
		# mat['specular'] = '{:.4f}'.format(np.random.rand())
		# mat['shininess'] = '{:.4f}'.format(np.random.rand())
		# mat['reflectance'] = '{:.4f}'.format(np.random.rand())
		# mat['emission'] = '{:.4f}'.format(np.random.rand())
		mat['texture'] = 'tex{}-{}'.format(oid,n)
		materials.append(mat)

		tex = {}
		tex['name'] = 'tex{}-{}'.format(oid,n)
		tex['builtin'] = 'flat'
		tex['height'] = '32'
		tex['width'] = '32'
		rgb = np.random.rand(3) * 0.7 + 0.3
		tex['rgb1'] = '{:.3f} {:.3f} {:.3f}'.format(*rgb)
		tex['type'] = 'cube'
		textures.append(tex)

	return geoms, materials, textures

def gen_skybox(use_floor=False): # [R, L, U, D, B, F]

	tex = {}

	rgb = [0.6, 0.3, 0.3] #np.random.rand(3) * 0.7 + 0.3

	tex['name'] = 'sky'
	tex['type'] = 'skybox'


	# tex['builtin'] = 'flat'
	# tex['height'] = '32'
	# tex['width'] = '32'
	# tex['rgb1'] = '{:.3f} {:.3f} {:.3f}'.format(*rgb)

	if use_floor:
		tex['fileright'] = '/home/fleeb/workspace/marl/foundation/mujoco_envs/noise.png'
		tex['fileleft'] = '/home/fleeb/workspace/marl/foundation/mujoco_envs/noise.png'
		tex['fileup'] = '/home/fleeb/workspace/marl/foundation/mujoco_envs/noise.png'
		tex['filedown'] = '/home/fleeb/workspace/marl/foundation/mujoco_envs/noise.png'
		tex['filefront'] = '/home/fleeb/workspace/marl/foundation/mujoco_envs/noise.png'
		tex['fileback'] = '/home/fleeb/workspace/marl/foundation/mujoco_envs/noise.png'

	else:

		# tex['file'] = np.random.choice(paths)


		#tex['file'] = '/home/fleeb/Downloads/Places365_val_00028035.png'
		tex['file'] = '/home/fleeb/workspace/marl/foundation/mujoco_envs/noise.png'

	return tex

def set_simple_texture(sim, idx, ):

	W, H = sim.model.tex_width[idx], sim.model.tex_height[idx]

	assert 6*W == H

	w = np.ones((W, W, 3), dtype=np.uint8)*255
	r = w * np.array([1, 0, 0]).reshape(1,1,-1)
	g = w * np.array([0, 1, 0]).reshape(1,1,-1)
	b = w * np.array([0, 0, 1]).reshape(1,1,-1)
	y = w * np.array([1, 1, 0]).reshape(1,1,-1)
	m = w * np.array([1, 0, 1]).reshape(1,1,-1)
	c = w * np.array([0, 1, 1]).reshape(1,1,-1)

	s = sim.model.tex_adr[idx]
	e = sim.model.tex_adr[idx + 1] if (idx + 1) < len(sim.model.tex_adr) else len(sim.model.tex_rgb)

	sim.model.tex_rgb[s:e] = np.concatenate([r,g,b,m,y,c], 0).reshape(-1)

def set_simple_shape(sim, idx=None, body_name='floating_obj', texture=True):

	if idx is None:
		idx = sim.model.body_name2id(body_name)

	N = sim.model.body_geomnum[idx]

	pos = np.zeros(3)
	quat = np.zeros(4)
	quat[-1] = 1

	shape = 6
	size = .12

	sel = sim.model.geom_bodyid == idx

	if texture:
		for n in np.arange(len(sim.model.geom_bodyid))[sel]:
			set_simple_texture(sim, n)

	mids = sim.model.geom_matid[sel]
	sim.model.mat_emission[mids] = 1
	sim.model.mat_specular[mids] = 0
	sim.model.mat_shininess[mids] = 0
	sim.model.mat_reflectance[mids] = 0

	sim.model.geom_type[sel] = shape
	sim.model.geom_pos[sel] = pos
	sim.model.geom_quat[sel] = quat

	sim.model.geom_size[sel] = size #[size] + [0.01]*(N-1)

# def sample_light():
# 	pass
#
# def randomize_light(sim, cam_name='external_camera_0', center=None, ):
# 	N = len(sim.model.light_active)

def randomize_obj(sim, body_name='floating_obj', mod_shape=True, mod_size=True, mod_mat=True,
				  shape_distrib=None, pos_mag=.1, size_mag=.08):
	bid = sim.model.body_name2id(body_name)

	N = sim.model.body_geomnum[bid]
	#shape = np.random.randint(5, size=(N,))+2 # 2=sphere, capsule, ellipsoid, cylinder, 6=box
	if shape_distrib is not None:
		shape_distrib = np.array(shape_distrib)
		shape_distrib = shape_distrib/shape_distrib.sum()
	shape = np.random.choice(5, size=N, p=shape_distrib) + 2
	size = np.random.rand(N, 3) * size_mag + size_mag

	quat = np.random.randn(N, 4)
	quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

	pos = np.random.randn(N, 3)
	pos /= np.linalg.norm(pos,axis=-1,keepdims=True)
	pos *= pos_mag
	pos -= pos.mean(0,keepdims=True)

	# print(pos)

	sel = sim.model.geom_bodyid == bid
	#print(sel, sim.model.geom_bodyid)

	if mod_mat:
		mids = sim.model.geom_matid[sel]
		sim.model.mat_emission[mids] = np.random.rand(N)
		sim.model.mat_specular[mids] = np.random.rand(N)
		sim.model.mat_shininess[mids] = np.random.rand(N)
		sim.model.mat_reflectance[mids] = np.random.rand(N)/4

	if mod_shape:
		sim.model.geom_type[sel] = shape
		sim.model.geom_pos[sel] = pos
		sim.model.geom_quat[sel] = quat

	if mod_size:
		sim.model.geom_size[sel] = size


def mod_textures(sim, texture_paths, bg_paths=None, bg_idx=None, use_floor=False):

	assert bg_idx is None or bg_paths is not None, 'no bg choices'

	W, H = sim.model.tex_width, sim.model.tex_height

	data = np.zeros(3*(W*H).sum(),dtype=np.uint8)

	ptr = 0
	for i, (h, w) in enumerate(zip(H,W)):
		N = h*w*3
		if bg_idx is not None and i == bg_idx:
			if use_floor: # [R, L, U, D, B, F]
				box = np.zeros((6,h//6,w,3))
				mh, mw = box.shape[1:3]
				box[:] = util.get_img(np.random.choice(bg_paths), mh, mw).reshape(1,mh,mw,3)
				box[:2] = box[0, :, ::-1]
				# box[1] = box[1, :, ::-1]
				# box[4] = box[0, :, ::-1]
				# box[5] = box[0, :, ::-1]
				box[2] = util.get_img(np.random.choice(texture_paths), mh, mw)
				box[3] = util.get_img(np.random.choice(texture_paths), mh, mw)
				data[ptr:ptr+N] = box.reshape(-1)
			else:
				data[ptr:ptr+N] = util.get_img(np.random.choice(bg_paths), h, w).reshape(-1)
		else:
			data[ptr:ptr+N] = util.get_patch(np.random.choice(texture_paths), h, w).reshape(-1)
		ptr += N

	sim.model.tex_rgb[:] = data.reshape(-1)


	# assert len(textures) == len(geoms)
	# for tex, idx in zip(textures, geoms):
	# 	s = sim.model.tex_adr[idx]
	# 	e = sim.model.tex_adr[idx+1] if (idx+1) < len(sim.model.tex_adr) else len(sim.model.tex_rgb)
	# 	sim.model.tex_rgb[s:e] = tex.reshape(-1)



