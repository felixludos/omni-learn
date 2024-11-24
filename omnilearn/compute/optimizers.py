from .imports import *
from ..abstract import AbstractMachine, AbstractOptimizer, AbstractModel, AbstractBatch
from ..machines import Machine
from .models import Model



class OptimizerBase(Machine, AbstractOptimizer):
	def __init__(self, *, objective: str = 'loss', maximize: bool = False, **kwargs):
		super().__init__(**kwargs)
		self._objective = objective
		self._maximize = maximize

	@property
	def objective(self) -> str:
		return self._objective
	
	@property
	def objective_direction(self) -> int:
		return 1 if self._maximize else -1
	

	def setup(self, model: AbstractModel) -> Self:
		return self



class PytorchOptimizer(OptimizerBase, O.Optimizer):
	def __init__(self, params=None, **kwargs):
		if params is None:
			params = [torch.zeros(0)]
		super().__init__(params=params, **kwargs)
		self.param_groups.clear()


	def add_parameters(self, parameters: Iterable['torch.nn.Parameter']) -> None:
		param_groups = list(parameters)
		if len(param_groups) == 0:
			raise ValueError("optimizer got an empty parameter list")
		if not isinstance(param_groups[0], dict):
			param_groups = [{'params': param_groups}]

		for param_group in param_groups:
			self.add_param_group(param_group)


	def setup(self, model: Model) -> Self:
		self.add_parameters(model.parameters())
		return super().setup(model)
		

	def step(self, batch: AbstractBatch) -> AbstractBatch:
		objective = batch[self.objective]
		if self._maximize:
			objective = -objective
		objective.backward()
		super(AbstractOptimizer, self).step()
		self.zero_grad()
		return batch


	def _checkpoint_data(self):
		data = {'settings': self.settings()}
		if self._is_prepared:
			state = self.state_dict()
			if state is not None and len(state):
				data['state_dict'] = state
		return data



class SGD(PytorchOptimizer, O.SGD):
	@property
	def name(self) -> str:
		lr = f'{self.defaults["lr"]:.0e}'.replace('+', '')
		return f'SGD{lr}'

	def settings(self):
		return {
			'lr': self.defaults['lr'],
			'momentum': self.defaults['momentum'],
			'dampening': self.defaults['dampening'],
			'weight_decay': self.defaults['weight_decay'],
			'nesterov': self.defaults['nesterov'],
		}

	def _load_checkpoint_data(self, data: Dict[str, Any], *, unsafe: bool = False) -> None:
		settings = data['settings']
		if self._is_prepared:
			current = self.settings()
			if current != settings:
				if unsafe:
					print(f'WARNING: settings do not match: {settings} != {current}')
				else:
					raise ValueError(f'settings do not match: {settings} != {current}')

		self.defaults['lr'] = settings['lr']
		self.defaults['momentum'] = settings['momentum']
		self.defaults['dampening'] = settings['dampening']
		self.defaults['weight_decay'] = settings['weight_decay']
		self.defaults['nesterov'] = settings['nesterov']

		if 'state_dict' in data:
			if not self._is_prepared:
				raise ValueError(f'optimizer must be prepared before loading state_dict')
			self.load_state_dict(data['state_dict'])



class Adam(PytorchOptimizer, O.Adam):
	@property
	def name(self) -> str:
		lr = f'{self.defaults["lr"]:.0e}'.replace('+', '')
		return f'Adam{lr}'

	def settings(self):
		return {
			'lr': self.defaults['lr'],
			'beta1': self.defaults['betas'][0],
			'beta2': self.defaults['betas'][1],
			'eps': self.defaults['eps'],
			'weight_decay': self.defaults['weight_decay'],
			'amsgrad': self.defaults['amsgrad'],
		}

	def _load_checkpoint_data(self, data: Dict[str, Any], *, unsafe: bool = False) -> None:
		settings = data['settings']
		if self._is_prepared:
			current = self.settings()
			if current != settings:
				if unsafe:
					print(f'WARNING: settings do not match: {settings} != {current}')
				else:
					raise ValueError(f'settings do not match: {settings} != {current}')

		self.defaults['lr'] = settings['lr']
		self.defaults['betas'] = (settings['beta1'], settings['beta2'])
		self.defaults['eps'] = settings['eps']
		self.defaults['weight_decay'] = settings['weight_decay']
		self.defaults['amsgrad'] = settings['amsgrad']

		if 'state_dict' in data:
			if not self._is_prepared:
				raise ValueError(f'optimizer must be prepared before loading state_dict')
			self.load_state_dict(data['state_dict'])


