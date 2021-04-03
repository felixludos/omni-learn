
try:
	import timm
except ImportError:
	print('WARNING: timm not found')
	pass
else:
	import omnifig as fig
	from wrapt import ObjectProxy
	import numpy as np
	from omnibelt import InitWall, unspecified_argument, Simple_Child
	from .framework import Function, FunctionWrapper
	
	@fig.Component('_pretrained')
	def build_pretrained(A, ID=None, pretrained=None, model_kwargs=None, include_config_args=None):
		
		if ID is None:
			ID = A.pull('model-ID', '<>ident')
		
		if pretrained is None:
			pretrained = A.pull('pretrai ned', True)
		
		if include_config_args is None:
			include_config_args = A.pull('include-config-args', True, silent=True)
		
		model_args = None
		if model_kwargs is None or include_config_args:
			model_args = A.pull('model-kwargs', '<>kwargs', {})
		
		if model_kwargs is not None:
			if model_args is None:
				model_args = model_kwargs
			else:
				model_args.update(model_kwargs)
		
		model = timm.create_model(ID, pretrained=pretrained, **model_args)
		model.identifier = ID
		model.pretrained = pretrained
		if pretrained:
			model.eval()
		return model
	
	@fig.Component('pretrained')
	class Pretrained(FunctionWrapper):
		def __init__(self, A, function=None, ID=None, pretrained=None, model_kwargs=None, include_config_args=None,
		             fine_tune=None,
		             **kwargs):
			
			if function is None:
				function = build_pretrained(A, ID=ID, pretrained=pretrained,
				                         include_config_args=include_config_args,
				                         model_kwargs=model_kwargs)
			
			if fine_tune is None:
				fine_tune = A.pull('fine-tune', True)
			
			if not fine_tune:
				for param in function.parameters():
					param.requires_grad = False
			
			super().__init__(A, function=function, **kwargs)
			
			self.fine_tune = fine_tune
			self.name = self.function.identifier
			self.pretrained = self.function.pretrained
			
		def extra_repr(self):
			return f'ident={self.name}, pretrained={self.pretrained}, fine-tune={self.fine_tune}'
			
	@fig.Component('feature-extractor')
	class FeatureExtractor(Pretrained):
		def __init__(self, A, feature_sel=unspecified_argument, model_kwargs=None,
		             features_only=unspecified_argument, reshape=None,
		             din=None, **kwargs):
			
			if feature_sel is unspecified_argument:
				feature_sel = A.pull('feature-sel', None)
			
			if features_only is unspecified_argument:
				features_only = A.pull('features-only', feature_sel is not None)
				
			if features_only:
				if model_kwargs is None:
					model_kwargs = {}
				model_kwargs['features_only'] = True
				
			if reshape is None:
				reshape = A.pull('reshape', False)
				
			if din is None:
				din = A.pull('din', None)
				
			super().__init__(A, din=din, model_kwargs=model_kwargs, **kwargs)
			
			if din is not None and feature_sel is not None:
				C, H, W = din
				
				info = self.feature_info[feature_sel]
				dout = info['num_chs'], H//info['reduction'], W//info['reduction']
				if reshape:
					dout = int(np.product(dout))
				self.dout = dout
				
			self.features_sel = feature_sel
			self.reshape = reshape

		def forward(self, x):
			
			out = super().forward(x)
			
			if self.features_sel is not None:
				out = out[self.features_sel]
				
				if self.reshape:
					B = out.size(0)
					out = out.reshape(B, -1)
			
			return out








