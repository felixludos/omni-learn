
import sys, os
import numpy as np
import torch
from omnibelt import get_printer

prt = get_printer(__name__)

try:
	import matplotlib.pyplot as plt
	from matplotlib import animation
	from matplotlib.path import Path
	import matplotlib.patches as patches
	from matplotlib.figure import figaspect
except ImportError:
	prt.warning('matplotlib not found')
try:
	import imageio
except ImportError:
	prt.warning('imageio not found')
# try:
# 	import ffmpeg
# except ImportError:
# 	print('WARNING: ffmpeg not found')
try:
	from IPython.display import HTML
except ImportError:
	prt.warning('ipython not found')

try:
	import pandas as pd
except ImportError:
	prt.warning('pandas not found')
try:
	import seaborn as sns
except ImportError:
	prt.warning('seaborn not found')

from torch.nn import functional as F

from .math import factors
from .data import load_images

def calc_tiling(N, H=None, W=None, prefer_tall=False):

	if H is not None and W is None:
		W = N//H
	if W is not None and H is None:
		H = N//W

	if H is not None and W is not None and N == H*W:
		return H, W

	H,W = tuple(factors(N))[-2:] # most middle 2 factors

	if H > W or prefer_tall:
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


def plot_vec_fn(f, din=None, dim=None, lim=(-5, 5), N=1000, figax=None):
	if din is None:
		din = f.din
	if dim is not None:
		raise NotImplementedError
	
	if figax is None:
		figax = plt.subplots()
	fig, ax = figax
	plt.sca(ax)
	
	vals = torch.linspace(*lim, N)
	with torch.no_grad():
		outs = f(vals.unsqueeze(-1).expand(-1, din))
	plt.plot(vals, outs)
	plt.xlabel('Input = [1,...,1] * x')
	plt.ylabel('Output')
	return fig, ax


def plot_hists(pts, figax=None, figsize=(6, 6), series_kwargs=None, sharex=False, **kwargs):
	
	if figax is None:
		figax = plt.subplots(len(pts), figsize=figsize, sharex=sharex)
	fig, axes = figax
	
	if series_kwargs is None:
		series_kwargs = [kwargs.copy() for _ in range(len(pts))]
	elif len(kwargs):
		series = []
		for s in series_kwargs:
			series.append(kwargs.copy())
			series[-1].update(s)
		series_kwargs = series
	
	for i, (ax, vals, kwargs) in enumerate(zip(axes, pts, series_kwargs)):
		plt.sca(ax)
		plt.hist(vals, **kwargs)
		
	return fig, axes


def plot_distribs(pts, figax=None, figsize=(6, 6), lim_y=None,
				  scale='count', inner='box', gridsize=100, cut=None, split=False,
				  color=None, palette=None, hue=None, **kwargs):
	
	if not isinstance(pts, list):
		pts = pts.tolist()
	
	inds = []
	vals = []
	for i, samples in enumerate(pts):
		inds.extend([i+1]*len(samples))
		vals.extend(samples)
	
	df = pd.DataFrame({'x': inds, 'y': vals})
	
	if figax is None:
		figax = plt.subplots(figsize=figsize)
	fig, ax = figax
	
	plt.sca(ax)
	
	# hue = None
	# split = False
	# color = 'C0'
	# inner = 'box'
	# palette = None
	
	if cut is not None:
		kwargs['cut'] = cut
	
	sns.violinplot(x='x', y='y', hue=hue,
				   data=df, split=split, color=color, palette=palette,
				   scale=scale, inner=inner, gridsize=gridsize, **kwargs)
	if lim_y is not None:
		plt.ylim(-lim_y, lim_y)
	# plt.title('Distributions of Latent Dimensions')
	# plt.xlabel('Dimension')
	# plt.ylabel('Values')
	# plt.tight_layout()
	# border, between = 0.02, 0.01
	# 	plt.subplots_adjust(wspace=between, hspace=between,
	# 						left=border, right=1 - border, bottom=border, top=1 - border)
	return fig, ax

def plot_parallel_coords(samples, categories=None, dim_names=None,
						 cat_styles=None, cat_names=None, include_legend=True,
						 mins=None, maxs=None,
						 figax=None, figsize=(8, 5), **default_style):
	'''
	samples: (N,D)
	categories: (N,)
	
	Example:

	from sklearn import datasets
	iris = datasets.load_iris()
	plot_parallel_coords(iris.data, dim_names=iris.feature_names,
						 categories=[iris.target_names[i] for i in iris.target])

	'''
	N, D = samples.shape
	
	classes = None
	if categories is not None:
		assert len(categories) == N, f'{len(categories)} vs {N}'
		
		classes = set(categories)
		if cat_names is not None:
			classes = cat_names
		K = len(classes)
		sep_classes = False
		if cat_styles is None:
			sep_classes = True
			cat_styles = {c: default_style.copy() for c in classes}
		
		for i, (name, style) in enumerate(cat_styles.items()):
			if 'color' in style and 'edgecolor' not in style:
				style['edgecolor'] = style['color']
				del style['color']
			elif sep_classes and 'edgecolor' not in style:
				style['edgecolor'] = f'C{i}'
			if 'facecolor' not in style:
				style['facecolor'] = 'none'
	
	try:
		samples = samples.cpu().numpy()
	except:
		pass
	
	if dim_names is None:
		dim_names = ['{}'.format(i) for i in range(D)]
	
	ynames = dim_names
	ys = samples
	
	if mins is None:
		mins = ys.min(axis=0)
	else:
		mins = mins.cpu().numpy()
	ymins = mins
	if maxs is None:
		maxs = ys.max(axis=0)
	else:
		maxs = maxs.cpu().numpy()
	ymaxs = maxs
	dys = ymaxs - ymins
	ymins -= dys * 0.05  # add 5% padding below and above
	ymaxs += dys * 0.05
	
	#     ymaxs[1], ymins[1] = ymins[1], ymaxs[1]  # reverse axis 1 to have less crossings
	dys = ymaxs - ymins
	
	# transform all data to be compatible with the main axis
	zs = np.zeros_like(ys)
	zs[:, 0] = ys[:, 0]
	zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
	
	if figax is None:
		figax = plt.subplots(figsize=figsize)
	fig, host = figax
	
	axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
	for i, ax in enumerate(axes):
		ax.set_ylim(ymins[i], ymaxs[i])
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		if ax != host:
			ax.spines['left'].set_visible(False)
			ax.yaxis.set_ticks_position('right')
			ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
	
	host.set_xlim(0, ys.shape[1] - 1)
	host.set_xticks(range(ys.shape[1]))
	host.set_xticklabels(ynames, fontsize=14)
	host.tick_params(axis='x', which='major', pad=7)
	host.spines['right'].set_visible(False)
	host.xaxis.tick_top()
	#     host.set_title('Parallel Coordinates Plot — Iris', fontsize=18, pad=12)
	
	#     colors = plt.cm.Set2.colors
	legend_handles = {}
	for j in range(ys.shape[0]):
		# create bezier curves
		verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
						 np.repeat(zs[j, :], 3)[1:-1]))
		codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
		path = Path(verts, codes)
		
		if categories is None:
			style = default_style
		else:
			cls = categories[j]
			if cat_names is not None:
				cls = cat_names[cls]
			style = cat_styles[cls]
		
		patch = patches.PathPatch(path, **style)  # facecolor='none', lw=2, alpha=0.7, edgecolor=colors[iris.target[j]])
		
		host.add_patch(patch)
		
		if categories is not None:
			legend_handles[cls] = patch
	
	if include_legend:
		host.legend([legend_handles[c] for c in classes], classes,
					loc='lower center', bbox_to_anchor=(0.5, -0.18),
					ncol=len(classes), fancybox=True, shadow=True)
	
	return fig, host
	

def plot_imgs(imgs, titles=None, H=None, W=None,
              figsize=None, scale=1,
              reverse_rows=False, grdlines=False,
              channel_first=None,
              imgroot=None, params={},
              savepath=None, dpi=96, autoclose=True, savescale=1,
              adjust={}, border=0., between=0.01):

	if isinstance(imgs, str):
		imgs = [imgs]
	if isinstance(imgs, (tuple, list)) and isinstance(imgs[0], str):
		imgs = list(load_images(*imgs, root=imgroot, channel_first=False))
		channel_first = False

	if isinstance(imgs, torch.Tensor):
		imgs = imgs.detach().cpu().squeeze(0).numpy()
	elif isinstance(imgs, (list, tuple)):
		imgs = [img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img for img in imgs]

	if isinstance(imgs, np.ndarray):
		shape = imgs.shape
		
		if channel_first is None \
				and shape[0 if len(shape) == 3 else 1] in {1, 3, 4} and shape[-1] not in {1, 3, 4}:
			channel_first = True
		
		# print(channel_first, shape)
		
		if len(shape) == 2 or (len(shape) == 3 and ((shape[0] in {1,3,4} and channel_first)
								  or (shape[-1] in {1,3,4} and not channel_first))):
			imgs = [imgs]
		elif len(shape) == 4:
			if channel_first:
				imgs = imgs.transpose(0,2,3,1)
				channel_first = False
		else:
			raise Exception(f'unknown shape: {shape}')

	imgs = [img.transpose(1,2,0).squeeze() if channel_first and len(img.shape)==3 else img.squeeze() for img in imgs]
	
	iH, iW = imgs[0].shape[:2]
	
	H,W = calc_tiling(len(imgs), H=H, W=W)
	
	fH, fW = iH*H, iW*W
	
	aw = None
	if figsize is None:
		aw, ah = figaspect(fH / fW)
		aw, ah = scale * aw, scale * ah
		figsize = aw, ah

	fg, axes = plt.subplots(H, W, figsize=figsize)
	axes = [axes] if len(imgs) == 1 else axes.flat

	hastitles = titles is not None and len(titles)
	if titles is None:
		titles = []
	if len(titles) != len(imgs):
		titles = titles + ([None] * (len(imgs) - len(titles)))

	for ax, img, title in zip(axes, imgs, titles):
		plt.sca(ax)
		if reverse_rows:
			img = img[::-1]
		plt.imshow(img, **params)
		if grdlines: # TODO: automate to allow more fine grain gridlines
			plt.plot([0, iW], [iH / 2, iH / 2], c='r', lw=.5, ls='--')
			plt.plot([iW / 2, iW / 2], [0, iH], c='r', lw=.5, ls='--')
			plt.xlim(0, iW)
			plt.ylim(0, iH)
			
		plt.axis('off')
		if title is not None:
			plt.title(title)
			
		# if title is not None:
		# 	plt.xticks([])
		# 	plt.yticks([])
		# 	plt.title(title)
		# else:
		# 	plt.axis('off')

	if hastitles and not len(adjust):
		plt.tight_layout()
	elif adjust is not None:
		
		base = dict(wspace=between, hspace=between,
							left=border, right=1 - border, bottom=border, top=1 - border)
		base.update(adjust)
		plt.subplots_adjust(**base)
	
	if savepath is not None:
		plt.savefig(savepath, dpi=savescale*(dpi if aw is None else fW/aw))
		if autoclose:
			plt.close()
			return
	
	return fg, axes


def plot_mat(M, val_fmt=None, figax=None, figsize=None, text_kwargs=dict(), **kwargs):
	H, W = M.shape
	if figax is None:
		figax = plt.subplots(figsize=figsize)
	fg, ax = figax
	
	plt.sca(ax)
	if isinstance(M, torch.Tensor):
		M = M.cpu().detach().numpy()
	if len(M.shape) == 1:
		M = M.reshape(1,-1)
	plt.matshow(M, False, **kwargs)
	eps = 0.03
	plt.subplots_adjust(eps,eps,1-eps,1-eps)
	if val_fmt is not None:
		if isinstance(val_fmt, int):
			val_fmt = f'.{val_fmt}f'
		if isinstance(val_fmt, str):
			val_fmt = '{:' + val_fmt + '}'
			fmt = lambda x: val_fmt.format(x)
		else:
			fmt = val_fmt
			
		if 'va' not in text_kwargs:
			text_kwargs['va'] = 'center'
		if 'ha' not in text_kwargs:
			text_kwargs['ha'] = 'center'
		for (i, j), z in np.ndenumerate(M):
			ax.text(j, i, fmt(z), **text_kwargs)
	return fg, ax


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

	def play(self, mode='mpl', **kwargs):

		if mode == 'mpl':
			play_back(self.frames)
		elif mode == 'jupyter':
			return HTML(self.as_animation(**kwargs).to_html5_video())
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


def image_size_limiter(imgs, size=(128,128)):
	H, W = imgs.shape[-2:]
	
	mH, mW = size
	
	if H <= mH and W <= mW:  # allows upto around 128x128
		return imgs
	
	imgs = F.interpolate(imgs, size=(128, 128))
	return imgs


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

