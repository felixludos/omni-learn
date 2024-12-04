from .imports import *
from ..core import *
from .. import spaces
from .models import Model

class Linear(Model, nn.Linear):
	def __init__(self, in_features: int = None, out_features: int = None, *, bias: bool = True, **kwargs):
		super().__init__(in_features=in_features or 1, out_features=out_features or 1, bias=bias, **kwargs)
		self.in_features = in_features
		self.out_features = out_features

	def _prepare(self, *, device = None):
		din = self.input_space.size
		dout = self.output_space.size
		if (dout, din) != self.weight.shape:
			if self.in_features is not None and self.in_features != din:
				print(f'WARNING: in_features was set to {self.in_features}, but preparing input size {din}')
			if self.out_features is not None and self.out_features != dout:
				print(f'WARNING: out_features was set to {self.out_features}, but preparing output size {dout}')
			self.weight = nn.Parameter(torch.empty(dout, din))
			self.in_features = din
			self.out_features = dout
			if self.bias is not None:
				self.bias = nn.Parameter(torch.empty(dout))
			self.reset_parameters()
		if device is not None:
			self.to(device)
		return super()._prepare(device=device)

	@space('input')
	def input_space(self) -> spaces.Vector:
		if self.in_features is None:
			raise self._GearFailed('in_features is not set')
		return spaces.Vector(self.in_features)
	
	@tool('output')
	def forward(self, input: 'torch.Tensor') -> 'torch.Tensor':
		return super().forward(input)
	@forward.space
	def output_space(self) -> spaces.Vector:
		if self.out_features is None:
			raise self._GearFailed('out_features is not set')
		return spaces.Vector(self.out_features)

	@property
	def name(self) -> str:
		return f'Linear({self.output_space.size}x{self.input_space.size}{"-nobias" if self.bias is None else ""})' if self._is_prepared \
			else (f'Linear({self.out_features}x{self.in_features}{"-nobias" if self.bias is None else ""})' 
		 			if self.in_features is not None and self.out_features is not None
						else f'Linear{"(nobias)" if self.bias is None else ""}')


class Mish(nn.Module):
	def forward(self, x):
		return x * torch.tanh(F.softplus(x))
class Swish(nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)



def get_nonlinearity(ident, dim=1, inplace=True, **kwargs):
	if ident is None:
		return None
	if not isinstance(ident, str):
		return ident

	if ident == 'prelu':
		return nn.PReLU(**kwargs)
	elif ident == 'lrelu':
		return nn.LeakyReLU(**kwargs)
	elif ident == 'relu':
		return nn.ReLU(inplace=inplace)
	elif ident == 'tanh':
		return nn.Tanh()
	elif ident == 'log-softmax':
		return nn.LogSoftmax(dim=dim)
	elif ident == 'softmax':
		return nn.Softmax(dim=dim)
	elif ident == 'softmax2d':
		return nn.Softmax2d()
	elif ident == 'softplus':
		return nn.Softplus(**kwargs)
	elif ident == 'sigmoid':
		return nn.Sigmoid()
	elif ident == 'elu':
		return nn.ELU(inplace=inplace, **kwargs)
	elif ident == 'selu':
		return nn.SELU(inplace=inplace, **kwargs)

	elif ident == 'mish':
		return Mish()
	elif ident == 'swish':
		return Swish()

	else:
		assert False, f'Unknown nonlinearity: {ident}'


