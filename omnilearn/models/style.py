

import torch

import omnifig as fig

from .nets import MultiLayer

from ..op.framework import Function
from .. import util

from .features import Prior, Gaussian, Uniform

class StyleFusionLayer(Function):
	def __init__(self, A, style_dim=None, **kwargs):

		if style_dim is None:
			style_dim = A.pull('style-dim', None)
			
		super().__init__(A, **kwargs)

		self.style_dim = style_dim
		self.register_buffer('_style', None)

	def get_style_dim(self):
		return self.style_dim

	def clear_style(self):
		self.cache_style(None)

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


class PriorStyleFusionLayer(Prior, StyleFusionLayer):

	def __init__(self, A, style_dim=None, **kwargs):
		super().__init__(A, style_dim=style_dim, prior_dim=style_dim, **kwargs)
		if style_dim is None:
			self.prior_dim = self.style_dim

	def forward(self, content, style=None, **kwargs):
		if style is None and self._style is None:
			self.cache_style(self.sample_prior(content.size(0)))
		return super().forward(content, style=style, **kwargs)


@fig.AutoModifier('gaussian-style')
class Gaussian(Gaussian, PriorStyleFusionLayer):
	pass

@fig.AutoModifier('uniform-style')
class Uniform(Uniform, PriorStyleFusionLayer):
	pass


class StyleExtractorLayer(Function):
	def __init__(self, A, style_dim=None, ret_content=None, ret_style=None, **kwargs):
		if style_dim is None:
			style_dim = A.pull('style-dim', None)

		if ret_content is None:
			ret_content = A.pull('ret_content', True)
		if ret_style is None:
			ret_style = A.pull('ret_style', False)

		super().__init__(A, **kwargs)

		self.style_dim = style_dim
		self.register_buffer('_style', None)
		self.register_buffer('_content', None)
		
		self._ret_style = ret_style
		self._ret_content = ret_content
		
	def get_style_dim(self):
		return self.style_dim
	
	def process_style(self, style):
		return style
	
	def process_content(self, content):
		return content
	
	def clear_cache(self, content=True, style=True):
		if content:
			self.cache_content()
		if style:
			self.cache_style()
	
	def cache_style(self, style=None):
		self._style = style
		
	def cache_content(self, content=None):
		self._content = content
		
	def collect_style(self):
		style = self._style
		self._style = None
		return style
	
	def extract(self, inp, **unused):
		raise NotImplementedError
	
	def forward(self, inp, ret_style=None, ret_content=None, **kwargs):
		if ret_style is None:
			ret_style = self._ret_style
		if ret_content is None:
			ret_content = self._ret_content
		
		content, style = self.extract(inp, **kwargs)
		
		content = self.process_content(content)
		style = self.process_style(style)

		if ret_content and ret_style:
			return content, style
		
		if ret_style:
			self.cache_content(content)
			return style
		
		self.cache_style(style)
		return content
		

class StyleMultiLayer(MultiLayer):
	
	def clear_style(self):
		super().clear_style()
		for layer in self.style_layers:
			layer.clear_style()


@fig.Component('style-fusion')
class StyleFusion(StyleFusionLayer, StyleMultiLayer):
	def __init__(self, A, **kwargs):

		split_style = A.pull('split-style', False)

		super().__init__(A, **kwargs)

		self.style_layers = [layer for layer in self.layers if isinstance(layer, StyleFusionLayer)]

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
		return super(StyleFusionLayer, self).forward(content)



@fig.Component('style-extractor')
class StyleExtractor(StyleExtractorLayer, StyleMultiLayer):
	def __init__(self, A, **kwargs):

		merge_style = A.pull('merge-style', True)
		
		super().__init__(A, **kwargs)
		
		self.style_layers = [layer for layer in self.layers if isinstance(layer, StyleExtractorLayer)]
		self.merge_style = merge_style
		
		
	def collect_style(self):
		styles = []
		for layer in self.style_layers:
			styles.append(layer.collect_style())
			
		if self.merge_style:
			styles = torch.cat(styles, 1)
		return styles
	
	
	def extract(self, inp, **unused):
		
		content = super(StyleExtractorLayer, self).forward(inp)
		style = self.collect_style()
		
		return content, style



