from .imports import *
from .abstract import AbstractModel, AbstractDataset



class ModelBase(nn.Module, AbstractModel):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._is_prepared = False


	def prepare(self, dataset: AbstractDataset, *, device: Optional[str] = None, **kwargs) -> Self:
		if self._is_prepared:
			return self
		self._prepare(dataset, device=device, **kwargs)
		self._is_prepared = True
		return self
	

	def _prepare(self, dataset: AbstractDataset, *, device: Optional[str] = None):
		pass
	


class MLP(ModelBase, nn.Sequential):
	def __init__(self, hidden: Optional[Iterable[int]] = None, *,
			  nonlin: str = 'elu', output_nonlin: Optional[str] = None, 
			  input_dim: Optional[int] = None, output_dim: Optional[int] = None,
			  **kwargs):
		super().__init__(**kwargs)
		self._hidden = hidden
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._nonlin = nonlin
		self._output_nonlin = output_nonlin


	def count_parameters(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad)


	@property
	def name(self) -> str:
		return f'MLP-{self._nonlin}' if not self._is_prepared else f'MLP-{self._nonlin}-{naturalsize(self.count_parameters(), gnu=True, format='%.0f').replace('G', 'M').replace('T', 'B')}'


	@staticmethod
	def _build(din: int, dout: int, hidden: Optional[Iterable[int]] = None,
			   nonlin: str = 'elu', output_nonlin: Optional[str] = None, 
			   initializer=None,):
		if hidden is None:
			hidden = []
		
		nonlins = [nonlin] * len(hidden) + [output_nonlin]
		hidden = din, *hidden, dout

		layers = []

		for in_dim, out_dim, nonlin in zip(hidden, hidden[1:], nonlins):
			layer = nn.Linear(in_dim, out_dim)
			if initializer is not None:
				layer = initializer(layer, nonlin)
			layers.append(layer)
			if nonlin is not None:
				layers.append(get_nonlinearity(nonlin))

		return layers


	def _prepare(self, dataset, *, device = None, 
			 input_dim: Optional[int] = None, output_dim: Optional[int] = None):
		if input_dim is None:
			input_dim = self._input_dim or dataset.input_dim
			if isinstance(input_dim, tuple):
				input_dim = int(np.prod(input_dim))
		if output_dim is None:
			output_dim = self._output_dim or dataset.output_dim

		self._input_dim = input_dim
		self._output_dim = output_dim
		layers = self._build(input_dim, output_dim, hidden=self._hidden, 
					   nonlin=self._nonlin, output_nonlin=self._output_nonlin)
		for i, layer in enumerate(layers):
			self.add_module(f'{i}', layer)
		
		if device is not None:
			self.to(device)
		return super()._prepare(dataset, device=device)



from torch.nn import functional as F


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

