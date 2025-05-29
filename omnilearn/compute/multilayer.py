from .imports import *
from ..core import *
from .. import spaces
from .math import parameter_count

from .models import Model
from .simple import get_nonlinearity, get_normalization_layer



class MLP(Model, nn.Sequential):
	def __init__(self, hidden: Optional[Iterable[int]] = None, *,
				 nonlin: str = 'elu', output_nonlin: Optional[str] = None,
				 norm: Optional[str] = None, dropout: Optional[float] = None,
				 output_norm: Optional[str] = None, output_dropout: Optional[float] = None,
				 input_dim: Optional[int] = None, output_dim: Optional[int] = None,
				 **kwargs):
		super().__init__(**kwargs)
		self._hidden = hidden
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._nonlin = nonlin
		self._output_nonlin = output_nonlin
		self._norm = norm
		self._dropout = dropout
		self._output_norm = output_norm
		self._output_dropout = output_dropout

	@space('input')
	def input_space(self) -> spaces.Vector:
		if self._input_dim is None:
			raise self._GearFailed('input_dim is not set')
		return spaces.Vector(self._input_dim) if isinstance(self._input_dim, int) \
			else self._input_dim

	@tool('output')
	def forward(self, input: 'torch.Tensor') -> 'torch.Tensor':
		return super().forward(input)

	@space('output')
	def output_space(self) -> spaces.Vector:
		if self._output_dim is None:
			raise self._GearFailed('output_dim is not set')
		return spaces.Vector(self._output_dim) if isinstance(self._output_dim, int) \
			else self._output_dim

	@property
	def name(self) -> str:
		# desc = f'{len(self._hidden)}-{self._nonlin}' if self._hidden is not None else f'-{self._nonlin}'
		# return f'MLP{desc}' if not self._is_prepared \
		# 	else f'MLP{desc}-{human_size(parameter_count(self))}'
		return 'mlp'

	def __str__(self):
		return super(AbstractGadget, self).__str__()

	def __repr__(self):
		return super(AbstractGadget, self).__repr__()

	@staticmethod
	def _build(din: Union[int, spaces.AbstractSpace], dout: Union[int, spaces.AbstractSpace],
			   hidden: Optional[Iterable[int]] = None, initializer=None,
			   nonlin: str = 'elu', output_nonlin: Optional[str] = None,
			   norm: Optional[str] = None, dropout: Optional[float] = None,
			   output_norm: Optional[str] = None, output_dropout: Optional[float] = None):
		if isinstance(din, spaces.AbstractSpace):
			din = din.size
		if isinstance(dout, spaces.AbstractSpace):
			dout = dout.size
		if hidden is None:
			hidden = []

		nonlins = [nonlin] * len(hidden) + [output_nonlin]
		norms = [norm] * len(hidden) + [output_norm]
		dropouts = [dropout] * len(hidden) + [output_dropout]
		hidden = din, *hidden, dout

		layers = []

		for in_dim, out_dim, nonlin, norm, dropout in zip(hidden, hidden[1:], nonlins, norms, dropouts):
			layer = nn.Linear(in_dim, out_dim)
			if initializer is not None:
				layer = initializer(layer, nonlin)
			layers.append(layer)
			if norm is not None:
				layers.append(get_normalization_layer(norm, out_dim))
			if nonlin is not None:
				layers.append(get_nonlinearity(nonlin))
			if dropout is not None:
				layers.append(nn.Dropout(dropout))

		return layers

	def setup(self, *, device: Optional[str] = None,
			  input_space: Optional[spaces.AbstractSpace] = None,
			  output_space: Optional[spaces.AbstractSpace] = None):
		input_space = input_space or self.input_space
		output_space = output_space or self.output_space
		layers = self._build(input_space, output_space, hidden=self._hidden,
							 nonlin=self._nonlin, output_nonlin=self._output_nonlin,
							 norm=self._norm, dropout=self._dropout,
							 output_norm=self._output_norm, output_dropout=self._output_dropout)
		for i, layer in enumerate(layers):
			self.add_module(f'{i}', layer)

		if device is not None:
			self.to(device)

	def settings(self):
		out = super().settings()
		out['hidden'] = list(self._hidden) if self._hidden is not None else None
		out['nonlin'] = self._nonlin
		out['output_nonlin'] = self._output_nonlin
		out['input_dim'] = self._input_dim
		out['output_dim'] = self._output_dim
		return out

	def _checkpoint_data(self):
		data = {'settings': self.settings()}
		if self._is_prepared:
			data['state_dict'] = self.state_dict()
		return data

	def _load_checkpoint_data(self, data: Dict[str, Any], *, unsafe: bool = False) -> None:
		settings = data['settings']
		if self._is_prepared:
			current = self.settings()
			if current != settings:
				if unsafe:
					print(f'WARNING: settings do not match: {settings} != {current}')
				else:
					raise ValueError(f'settings do not match: {settings} != {current}')

		if 'hidden' in settings:
			self._hidden = settings['hidden']
		if 'nonlin' in settings:
			self._nonlin = settings['nonlin']
		if 'output_nonlin' in settings:
			self._output_nonlin = settings['output_nonlin']
		if 'input_dim' in settings:
			self._input_dim = settings['input_dim']
		if 'output_dim' in settings:
			self._output_dim = settings['output_dim']
		if 'state_dict' in data:
			self.prepare()
			self.load_state_dict(data['state_dict'])




