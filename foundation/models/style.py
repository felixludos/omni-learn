

import torch

import omnifig as fig

from .nets import MultiLayer

from ..op.framework import Function
from .. import util

from .features import Prior, Gaussian, Uniform


class StyleLayer(Function):

	def __init__(self, A, style_dim=None, **kwargs):

		if style_dim is None:
			style_dim = A.pull('style-dim', None)

		super().__init__(A, **kwargs)

		self.style_dim = style_dim
		self.register_buffer('_style', None)

	def get_style_dim(self):
		return self.style_dim

	def cache_style(self, style):
		if self._style is None:
			self._style = style

	def process_style(self, style):

		if style is None:
			assert self._style is not None, 'no style provided or cached'
			style = self._style
		elif self._style is not None:
			raise Exception('Provided a style, when one was already cached')

		self._style = None

		return style
	
	def process_content(self, content):
		return content

	def infuse(self, content, style, **kwargs):
		raise NotImplementedError

	def forward(self, content, style=None, **kwargs):

		style = self.process_style(style)
		content = self.process_content(content)

		return self.infuse(content, style, **kwargs)

class PriorStyleLayer(Prior, StyleLayer):

	def __init__(self, A, style_dim=None, **kwargs):
		super().__init__(A, style_dim=style_dim, prior_dim=style_dim, **kwargs)
		if style_dim is None:
			self.prior_dim = self.style_dim

	def forward(self, content, style=None, **kwargs):
		if style is None and self._style is None:
			self.cache_style(self.sample_prior(content.size(0)))
		return super().forward(content, style=style, **kwargs)

@fig.AutoModifier('gaussian-style')
class Gaussian(Gaussian, PriorStyleLayer):
	pass

@fig.AutoModifier('uniform-style')
class Uniform(Uniform, PriorStyleLayer):
	pass


@fig.Component('style-multilayer')
class StyleSharing(StyleLayer, MultiLayer):

	def __init__(self, A, **kwargs):

		split_style = A.pull('split-style', False)

		super().__init__(A, **kwargs)

		self.style_layers = [layer for layer in self.layers if isinstance(layer, StyleLayer)]

		style_dims = None
		if split_style:
			style_dims = [layer.get_style_dim() for layer in self.style_layers]
			style_dim = sum(style_dims)
		else:
			style_dim = self.style_layers[0].get_style_dim()
			for layer in self.style_layers:
				assert layer.get_style_dim() == style_dim

		self.style_dims = style_dims
		self.style_dim = style_dim

	def share_style(self, style):
		styles = [style] * len(self.style_layers) if self.style_dims is None else style.split(self.style_dims, dim=1)
		for layer, style in zip(self.style_layers, styles):
			layer.cache_style(style)
		return styles

	def infuse(self, content, style, **unused):

		self.share_style(style)
		
		# run forward pass through all layers starting with "root"
		return super(StyleLayer, self).forward(content)



