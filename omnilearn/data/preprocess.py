
import sys, os
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import omnifig as fig

from .. import util


@fig.AutoComponent('flip')
class RandomFlip(nn.Module):

	def __init__(self, horizontal=None, vertical=None):
		super().__init__()

		self.horizontal = horizontal
		assert horizontal is None or 0 <= horizontal <= 1
		self.vertical = vertical
		assert vertical is None or 0 <= vertical <= 1

	def forward(self, imgs):

		if self.horizontal is not None and self.horizontal > 0 and np.random.rand() < self.horizontal:
			# print('h')
			imgs = imgs.flip(-1)

		if self.vertical is not None and self.vertical > 0 and np.random.rand() < self.vertical:
			# print('v')
			imgs = imgs.flip(-2)

		return imgs


@fig.AutoComponent('random-orientation')
class RandomOrientation(nn.Module):

	def forward(self, imgs, x=None):

		if x is None:
			x = np.random.randint(8)

		if x < 4:
			# print('h')
			imgs = imgs.flip(-1) # horizontal pos
			if x < 2:
				# print('t')
				imgs = imgs.transpose(-1, -2)
				if x < 1:
					# print('h')
					imgs = imgs.flip(-1)  # horizontal pos
			elif x < 3:
				# print('v')
				imgs = imgs.flip(-2)  # vertical pos

		elif x < 6:
			# print('v')
			imgs = imgs.flip(-2)  # vertical pos
			if x < 5:
				# print('t')
				imgs = imgs.transpose(-1, -2)

		elif x < 7:
			# print('t')
			imgs = imgs.transpose(-1, -2)

		# else:
		# print('i')

		return imgs

@fig.AutoComponent('image-transform')
class Image_Transform(nn.Module):
	def __init__(self, prob=None,
	             flip_h=False, flip_w=False, rot_90=False,
	             rot=False, scale=None, offset=None, flip=None,
	             brightness=None, contrast=None, noise=None,
	             src_bg=True):

		super().__init__()

		if not flip_h and not flip_w and rot_90:
			raise NotImplementedError

		num = 0

		num += int(flip_h)

		num += int(flip_w)

		self.flip_h = flip_h
		self.flip_w = flip_w

		num += int(rot)
		contin = rot

		if scale is not None and scale > 0:
			num += 1
			contin = True

		if offset is not None and offset > 0:
			num += 1
			contin = True

		if contin and (flip_h or flip_w) and flip is None:
			flip = 0.5

		if brightness is not None and brightness > 0:
			num += 1

		if contrast is not None and contrast > 0:
			num += 1

		if noise is not None and noise > 0:
			num += 1

		if prob is not None:
			assert 0 < prob < 1
		self.prob = prob

		if num == 0:
			print('WARNING: no transforms selected')

		self.num = num
		self.contin = contin
		self.src_bg = src_bg
		self.orient = RandomOrientation() if flip_w and flip_h and rot_90 else None
		if self.orient is None and rot_90:
			raise NotImplementedError

		if flip is None and (flip_h or flip_w):
			flip = 0.5

		self.rot = rot
		assert scale is None or 0<scale<2, f'val: {scale}'
		self.scale = scale
		assert offset is None or 0<offset<2, f'val: {offset}'
		self.offset = offset

		assert flip is None or 0<=flip<=1, f'val: {flip}'
		self.flip = flip

		assert brightness is None or 0<=brightness<=1, f'val: {brightness}'
		self.brightness = brightness
		assert contrast is None or 0<=contrast<=1, f'val: {contrast}'
		self.contrast = contrast
		assert noise is None or 0<=noise<=1, f'val: {noise}'
		self.noise = noise

	def __len__(self):
		return self.num

	def gen_vals(self, mag, N, device='cpu'):
		if mag is None or mag <= 0:
			return torch.ones(N, device=device)
		return torch.rand(N, device=device)*2*mag + 1 - mag

	def forward(self, x):

		if len(self) and (self.prob is None or np.random.rand() < self.prob):

			B, C, H, W = x.size()

			if self.contin:

				if self.src_bg:
					alpha = torch.ones(B, 1, H, W, device=x.device)
					x = torch.cat([x, alpha], 1)
					initial = x

				thetas = torch.rand(B) * 2 * np.pi if self.rot else torch.zeros(B)
				xy = self.gen_vals(self.offset, B*2).sub(1).view(B, 2, 1)
				scales = self.gen_vals(self.scale, B).view(B,1,1)

				se2s = util.se2_tfm(util.rots_2d(thetas)*scales, xy).to(x.device)

				x = F.grid_sample(x, F.affine_grid(se2s, x.size()))

				if self.src_bg:
					mask = (x[:, -1:] > 0.99).float()
					x = x * mask + initial * (1 - mask)
					x = x[:, :-1]

				if self.flip is not None and self.flip > 0:
					sel = (torch.rand(B, 1, 1, 1) < self.flip).float().to(x.device)
					x = sel * x.flip(-1) + (1 - sel) * x

			elif self.orient is not None:
				x = self.orient(x)

			elif self.flip_h or self.flip_w:
				if self.flip_h and self.flip_w:
					v = np.random.rand()
					if v < .6666:
						x = x.flip(-1)
					if v > .3333:
						x = x.flip(-2)
				else:
					x = x.flip(-1 if self.flip_h else -2)

			if self.contrast is not None and self.contrast > 0:
				contrast = self.gen_vals(self.contrast, B, device=x.device).view(B,1,1,1)

				x = contrast * x + (1 - contrast) * x.mean(1, keepdim=True)

			if self.brightness is not None and self.brightness > 0:
				brightness = self.gen_vals(self.brightness, B, device=x.device).view(B,1,1,1)

				x = brightness * x

			if self.noise is not None and self.noise > 0:
				x = x + torch.randn_like(x)*self.noise

			x = x.clamp(min=0, max=1)

		return x

