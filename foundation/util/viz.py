
import sys, os
import numpy as np
import torch
try:
	import matplotlib.pyplot as plt
	from matplotlib import animation
except ImportError:
	print('WARNING: matplotlib not found')
try:
	import imageio
except ImportError:
	print('WARNING: imageio not found')
# try:
# 	import ffmpeg
# except ImportError:
# 	print('WARNING: ffmpeg not found')
try:
	from IPython.display import HTML
except ImportError:
	print('WARNING: ipython not found')

from .math import factors

def calc_tiling(N, H=None, W=None, prefer_tall=False):

	if H is not None and W is None:
		W = N//H
	if W is not None and H is None:
		H = N//W

	if H is not None and W is not None and N == H*W:
		return H, W

	H,W = tuple(factors(N))[-2:] # most middle 2 factors

	if H > W and not prefer_tall:
		H, W = W, H

	# if not prefer_tall:
	# 	H,W = W,H
	return H, W


def tile_imgs(imgs, dim=0, H=None, W=None): # for numpy images

	raise NotImplementedError

	assert len(imgs.shape) > 3, 'not enough dimensions'
	if dim != 0:
		raise NotImplementedError

	N, h, w, c = imgs.shape

	if H is None or W is None:

		if H is None:
			W = 0

		pass

	assert H*W == imgs.shape[dim], 'Invalid tiling'


def show_nums(imgs, titles=None, H=None, W=None, figsize=(6, 6),
			  reverse_rows=False, grdlines=False, tight=False,
			  border=0.02, between=0.01):
	H,W = calc_tiling(imgs.size(0), H=H, W=W)

	imgs = imgs.cpu().permute(0, 2, 3, 1).squeeze().numpy()

	fig, axes = plt.subplots(H, W, figsize=figsize)

	if titles is None:
		titles = [None] * len(imgs)

	iH, iW = imgs.shape[1], imgs.shape[2]

	for ax, img, title in zip(axes.flat, imgs, titles):
		plt.sca(ax)
		if reverse_rows:
			img = img[::-1]
		plt.imshow(img)
		if grdlines:
			plt.plot([0, iW], [iH / 2, iH / 2], c='r', lw=.5, ls='--')
			plt.plot([iW / 2, iW / 2], [0, iH], c='r', lw=.5, ls='--')
			plt.xlim(0, iW)
			plt.ylim(0, iH)
		if title is not None:
			plt.xticks([])
			plt.yticks([])
			plt.title(title)
		else:
			plt.axis('off')

	if tight:
		plt.subplots_adjust(wspace=between, hspace=between,
							left=border, right=1 - border, bottom=border, top=1 - border)

	return fig, axes



def play_back(imgs, figax=None, batch_first=True): # imgs is numpy: either (seq, H, W, C) or (batch, seq, H, W, C)
	if len(imgs.shape) > 4:  # tile first
		if not batch_first:
			imgs = imgs.transpose(1,0,2,3,4)
		N, S, h, w, c = imgs.shape
		H, W = calc_tiling(N)
		imgs = imgs.reshape(H, W, S, h, w, c)
		imgs = imgs.transpose(2, 0, 3, 1, 4, 5)
		imgs = imgs.reshape(S, H * h, W * w, c)

	if figax is None:
		figax = plt.subplots()
	fig, ax = figax
	plt.sca(ax)
	for i, img in enumerate(imgs):
		plt.cla()
		plt.axis('off')
		plt.imshow(img)
		plt.title('{}/{}'.format(i + 1, len(imgs)))
		plt.tight_layout()
		plt.pause(0.02)
	return fig, ax

def plot_stat(all_stats, key, xdata=None, figax=None, **fmt_args): # all_stats should be a list of StatsMeters containing key
	if figax is None:
		figax = plt.subplots()
	fig, ax = figax

	if xdata is not None:
		assert len(all_stats) == len(xdata)
	else:
		xdata = np.arange(1, len(all_stats)+1)

	ydata = np.array([stat[key].avg.item() for stat in all_stats])

	plt.sca(ax)
	plt.plot(xdata, ydata, **fmt_args)

	return figax

def fig_to_rgba(fig):
	fig.canvas.draw()
	w, h = fig.canvas.get_width_height()
	buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
	buf.shape = (h, w, 4)
	buf = np.roll(buf, 3, axis=2)
	return buf

def flatten_tree(tree, is_tree=None, prefix=None): # returns tuples of deep keys
	if prefix is None:
		prefix = []

	flat = []

	for k, v in tree.items():
		prefix.append(k)
		if (is_tree is not None and is_tree(v)) or isinstance(v, dict):
			new = flatten_tree(v, is_tree=is_tree, prefix=prefix)
			if len(new):
				flat.extend(new)
				prefix.pop()
				continue
		flat.append(tuple(prefix))
		prefix.pop()

	return flat

class Video(object):
	def __init__(self, frames): # frames must be a sequence of numpy byte images
		self.frames = frames
		# self.path = path

	def play(self, mode='mpl', scale=1):

		if mode == 'mpl':
			play_back(self.frames)
		elif mode == 'jupyter':
			return HTML(self.as_animation(scale=scale).to_html5_video())
		else:
			raise Exception('Unknonwn mode: {}'.format(mode))

	def as_animation(self, scale=1, fps=20):

		H, W, C = self.frames[0].shape

		asp = W/H

		fig = plt.figure(figsize=(asp, 1), dpi=int(H*scale),)

		ax = plt.axes([0, 0, 1, 1], frameon=False)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		plt.autoscale(tight=True)

		im = plt.imshow(self.frames[0])
		# plt.axis('off')
		# plt.tight_layout()

		plt.close()

		def init():
			im.set_data(self.frames[0])

		def animate(i):
			im.set_data(self.frames[i])
			return im

		anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.frames), interval=1000//fps)

		return anim

	def export(self, path, fmt='mp4'):

		assert fmt in {'mp4', 'gif', 'jpg', 'png'}

		if fmt in {'mp4', 'gif'}:
			imageio.mimsave(path, self.frames)

		elif fmt in {'jpg', 'png'}:
			raise NotImplementedError

		# elif fmt == 'mp4':
		# 	imageio.mimwrite('test2.mp4', self.frames, fps=30)

			# vcodec = 'mp4'
			# framerate = 25
			#
			# images = self.frames
			# n, height, width, channels = images.shape
			# process = (
			# 	ffmpeg
			# 		.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
			# 		.output(path, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
			# 		.overwrite_output()
			# 		.run_async(pipe_stdin=True)
			# )
			# for frame in images:
			# 	process.stdin.write(
			# 		frame
			# 			.astype(np.uint8)
			# 			.tobytes()
			# 	)
			# process.stdin.close()
			# process.wait()

class Animation(Video):

	def __init__(self, anim):
		super().__init__(None) # no frames
		self.anim = anim

	def play(self, mode='jupyter', scale=1):

		if mode != 'jupyter':
			raise NotImplementedError

		if scale != 1:
			print('WARNING: scale has no effect for animations that are already done')

		return HTML(self.anim.to_html5_video())

	def as_animation(self, scale=1, fps=None):
		if scale != 1 or fps is not None:
			print('WARNING: scale/fps has no effect for animations that are already done')
		return self.anim

	def export(self, path, fmt=None, fps=20):

		assert not os.path.isdir(path), 'path is already a dir'

		if fmt is None:
			fmt = os.path.basename(path).split('.')[-1]
			if fmt in {'png', 'jpg'}:
				fmt = 'frames'

		if fmt == 'mp4':
			Writer = animation.writers['ffmpeg']
			writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
			self.anim.save(path, writer=writer)

		elif fmt == 'gif':
			self.anim.save(path, writer='imagemagick', fps=fps)

		elif fmt == 'frames':
			self.anim.save(path, writer="imagemagick")


class Pretty_Formatter(object):
	def __init__(self, lf='\n', ht='  '):
		self.types = {}
		self.lf, self.ht = lf, ht
		self.reset()

	def set_formater(self, typ, callback):
		self.types[typ] = callback

	def reset(self):
		self.types = {
			object: self.format_object,
			dict: self.format_dict,
			list: self.make_iter_format(['['], [']']),
			set: self.make_iter_format(['{'], ['}']),
			tuple: self.make_iter_format(['('], [')']),
		}

	def _indent(self, num=None):
		return self.lf + self.ht*num

	def __call__(self, value, formatters={}, lf=None, ht=None):

		self.types.update(formatters)

		if lf is not None:
			self.lf = lf
		if ht is not None:
			self.ht = ht

		deep_lines = self._format(value)

		s = self._flatten(deep_lines)
		if s[0] == self.lf:
			s = s[1:]
		return s

	def _format(self, value):
		out = None
		for cls in value.__class__.__mro__:
			if cls in self.types:
				out = self.types[cls](value, self._format) # should return a list of tuples/strings/lists
				break

		if out is None:
			print(f'WARNING: unable to format {value}')

		return out

	def _flatten(self, line, indent=0):

		if type(line) == str:
			return line.replace(self.lf, self._indent(indent+2))

		if type(line) == tuple:
			return ''.join(self._flatten(w, indent=indent) for w in line)

		if type(line) == list:
			ind = self._indent(indent+1)
			lines = [self._flatten((l,) if type(l) == str else l, indent=indent+1)
			                for l in line]
			start = ind.join(lines[:-1])
			ret = self._indent(indent)
			start += ret + lines[-1]
			return start

		raise Exception(f'not recognized: {line}')


	@staticmethod
	def format_object(value, fmt):
		return repr(value)

	@staticmethod
	def format_dict(value, fmt):
		lines = []
		lines.append('{')
		lines.extend((repr(key),': ', fmt(val)) for key, val in value.items())
		# lines.extend((repr(key), ': ', fmt(val), ',') for key, val in value.items())
		lines.append('}')
		return lines

	@staticmethod
	def make_iter_format(start, end):
		def _iter_format(value, fmt):
			nonlocal start, end
			lines = []
			lines.extend(start)
			lines.extend((fmt(val)) for val in value)
			# lines.extend((fmt(val), ',') for val in value)
			lines.extend(end)
			return lines
		return _iter_format

pretty_format = Pretty_Formatter()



###################
# Flow visualization - using yiq

_yiq_to_rgb = torch.FloatTensor([[1, 0.956, 0.621],
						[1, -0.272, -0.647],
						[1, -1.106, 1.703]])
_iq_to_rgb = torch.FloatTensor([[0.956, 0.621],
						[-0.272, -0.647],
						[-1.106, 1.703]])
_rgb_to_yiq = torch.FloatTensor([[.299, .587, 0.114],
						   [0.596, -0.274, -0.322],
						   [0.211, -0.523, 0.312]])

# for 2d flow
def iq2rgb(img, clamp=1): # reqs B x C=2 x H x W
	B, C, H, W = img.size()
	assert C == 2
	assert clamp > 0
	#print(img.permute(1,0,2,3).contiguous().view(2, -1).contiguous().clamp(-clamp, clamp).size())
	#quit()
	img = img.permute(1,0,2,3).contiguous().view(2, -1).contiguous().clamp(-clamp, clamp) * torch.Tensor([[0.5957], [0.5226]]).type_as(img) / clamp
	#print(img.abs().max(-1)[0])
	img = _iq_to_rgb.type_as(img) @ img + 0.5
	return img.view(3,B,H,W).permute(1,0,2,3).clamp(0,1)

# for 3d flow
def yiq2rgb(img, clamp=1): # reqs B x C=3 x H x W
	B, C, H, W = img.size()
	assert C == 3
	assert clamp > 0
	#print(img.permute(1,0,2,3).contiguous().view(2, -1).contiguous().clamp(-clamp, clamp).size())
	#quit()
	img = img.permute(1,0,2,3).contiguous().view(3, -1).contiguous().clamp(-clamp, clamp) * torch.Tensor([[0.5957], [0.5226], [1]]).type_as(img) / clamp
	#print(img.abs().max(-1)[0])
	img = _yiq_to_rgb.type_as(img) @ img + 0.5
	return img.view(3,B,H,W).permute(1,0,2,3).clamp(0,1)





# def tile_images(img_nhwc):
#     """
#     Tile N images into one big PxQ image
#     (P,Q) are chosen to be as close as possible, and if N
#     is square, then P=Q.
#
#     input: img_nhwc, list or array of images, ndim=4 once turned into array
#         n = batch index, h = height, w = width, c = channel
#     returns:
#         bigim_HWc, ndarray with ndim=3
#     """
#     img_nhwc = np.asarray(img_nhwc)
#     N, h, w, c = img_nhwc.shape
#     H = int(np.ceil(np.sqrt(N)))
#     W = int(np.ceil(float(N)/H))
#     img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
#     img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
#     img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
#     img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
#     return img_Hh_Ww_c

