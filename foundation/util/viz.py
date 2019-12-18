
import sys, os
import numpy as np
import torch
import matplotlib.pyplot as plt


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
