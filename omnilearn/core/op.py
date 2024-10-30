from .imports import *
import omnifig as fig

from .abstract import AbstractModel, AbstractOptimizer, AbstractEvent, AbstractMachine
from .planning import DefaultPlanner as Planner
from .datasets import FileDatasetBase, Batch
from .models import MLP as MLPBase, ModelBase
from .trainers import CheckpointableTrainer
from .optimizers import Adam as AdamBase, SGD as SGDBase, OptimizerBase as Optimizer
from .events import Pbar_Reporter, Checkpointer as CheckpointerBase

# configurable versions of the top level functions



class Machine(fig.Configurable, ToolKit, AbstractMachine):
	@fig.config_aliases(gap='app')
	def __init__(self, gap=None, **kwargs):
		super().__init__(gap=gap, **kwargs)


	def _checkpoint_data(self) -> Dict[str, Any]:
		return self.settings()


	def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
		assert data == self.settings(), 'checkpoint data does not match settings'
		raise NotImplementedError


	def checkpoint(self, path: Path = None) -> Any:
		data = self._checkpoint_data()
		if data is None or not len(data):
			return None
		if path is None:
			return data
		if path.suffix == '': path = path.with_suffix('.pt')
		torch.save(data, path)
		return path
	

	def load_checkpoint(self, *, path = None, data = None) -> None:
		assert path is None != data is None, 'must provide exactly one of path or data'
		if data is None:
			if path.suffix == '': path = path.with_suffix('.pt')
			assert path.exists(), f'checkpoint file does not exist: {path}'
			data = torch.load(path)
		self._load_checkpoint_data(data)
		return path


	def settings(self) -> Dict[str, Any]:
		return {}
	

	def indicators(self) -> Iterator[str]:
		yield from ()


class Reporter(Machine, Pbar_Reporter):
	@fig.config_aliases(print_metrics='log')
	def __init__(self, print_metrics: Iterable[str] = None, **kwargs):
		super().__init__(print_metrics=print_metrics, **kwargs)


class Checkpointer(Machine, CheckpointerBase):
	def __init__(self, saveroot: Path = 'checkpoints', *, freq: int = None, skip_0: bool = True, **kwargs):
		if saveroot is None:
			print(f'WARNING: No saveroot provided, so no checkpoints will be saved')
			freq = None
		else:
			saveroot = Path(saveroot)
		super().__init__(saveroot=saveroot, freq=freq, skip_0=skip_0, **kwargs)


class Trainer(fig.Configurable, CheckpointableTrainer):
	_Planner = Planner
	_Reporter = Reporter
	def __init__(self, model: AbstractModel, optimizer: AbstractOptimizer, *, 
			  reporter: AbstractEvent = None, env: Dict[str, AbstractMachine] = None, 
			  indicators: Iterable[str] = None, 
			  budget: Union[int, Dict[str, int]] = None, batch_size: int = None,
			  device: str = None, **kwargs):
		if indicators is None:
			indicators = []
		if isinstance(budget, int):
			budget = {'max_iterations': budget}
		super().__init__(model=model, optimizer=optimizer, reporter=reporter, env=env, batch_size=batch_size, device=device, **kwargs)
		if budget is not None:
			self._planner.budget(**budget)
		self._indicators = indicators


	def all_indicators(self) -> Iterator[str]:
		past = set()
		for key in self._indicators:
			if key not in past:
				past.add(key)
				yield key
		for key in self._model.indicators():
			if key not in past:
				past.add(key)
				yield key
		for key in self._optimizer.indicators():
			if key not in past:
				past.add(key)
				yield key
		for machine in self._env.values():
			for key in machine.indicators():
				if key not in past:
					past.add(key)
					yield key



class Dataset(Machine, FileDatasetBase):
	def __iter__(self) -> Iterator[Batch]:
		return self.iterate()
	

	def __len__(self) -> int:
		return self.size



class Model(Machine, ModelBase):
	pass



class MLP(Model, MLPBase):
	def __init__(self, hidden: Optional[Iterable[int]] = None, *,
				 nonlin: str = 'elu', output_nonlin: Optional[str] = None,
				 input_dim: Optional[int] = None, output_dim: Optional[int] = None,
				 **kwargs):
		super().__init__(hidden=hidden, nonlin=nonlin, output_nonlin=output_nonlin, input_dim=input_dim, output_dim=output_dim, **kwargs)


	@tool('output')
	def __call__(self, input: torch.Tensor) -> torch.Tensor:
		return super().__call__(input)
	

	def settings(self):
		out = super().settings()
		out['hidden'] = self._hidden
		out['nonlin'] = self._nonlin
		out['output_nonlin'] = self._output_nonlin
		out['input_dim'] = self._input_dim
		out['output_dim'] = self._output_dim
		return out
	
			
	def _checkpoint_data(self):
		data = {'settings': super().settings()}
		if self._is_prepared:
			data['state_dict'] = self.state_dict()
		return data


	def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
		settings = data['settings']
		if self._is_prepared:
			current = self.settings()
			if current != settings:
				error_keys = [key for key in settings if settings.get(key) != current.get(key)]
				errors = ', '.join([f'{key}: {settings[key]} != {current[key]}' for key in error_keys])
				raise ValueError(f'settings do not match: {errors}')
			
		self._hidden = settings['hidden']
		self._nonlin = settings['nonlin']
		self._output_nonlin = settings['output_nonlin']
		self._input_dim = settings['input_dim']
		self._output_dim = settings['output_dim']

		if 'state_dict' in data:
			if not self._is_prepared:
				self.prepare(None)
			self.load_state_dict(data['state_dict'])




class SGD(Machine, SGDBase):
	def __init__(self, lr: float, momentum: float = 0., dampening: float = 0., 
				 weight_decay: float = 0., nesterov: bool = False, **kwargs):
		super().__init__(lr=lr, momentum=momentum, dampening=dampening, 
						 weight_decay=weight_decay, nesterov=nesterov, **kwargs)


	def settings(self):
		return {
			'lr': self.defaults['lr'],
			'momentum': self.defaults['momentum'],
			'dampening': self.defaults['dampening'],
			'weight_decay': self.defaults['weight_decay'],
			'nesterov': self.defaults['nesterov'],
		}


	def _checkpoint_data(self):
		data = {'settings': super().settings()}
		data['state_dict'] = self.state_dict()
		return data


	def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
		settings = data['settings']
		if self._is_prepared:
			current = self.settings()
			if current != settings:
				error_keys = [key for key in settings if settings.get(key) != current.get(key)]
				errors = ', '.join([f'{key}: {settings[key]} != {current[key]}' for key in error_keys])
				raise ValueError(f'settings do not match: {errors}')
			
		self.defaults['lr'] = settings['lr']
		self.defaults['momentum'] = settings['momentum']
		self.defaults['dampening'] = settings['dampening']
		self.defaults['weight_decay'] = settings['weight_decay']
		self.defaults['nesterov'] = settings['nesterov']

		if 'state_dict' in data:
			if not self._is_prepared:
				self.prepare(None)
			self.load_state_dict(data['state_dict'])



class Adam(Machine, AdamBase):
	def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
				 eps: float = 1e-8, weight_decay: float = 0., amsgrad: bool = False, 
				 **kwargs):
		super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, 
						 amsgrad=amsgrad, **kwargs)


	def settings(self):
		return {
			'lr': self.defaults['lr'],
			'beta1': self.defaults['betas'][0],
			'beta2': self.defaults['betas'][1],
			'eps': self.defaults['eps'],
			'weight_decay': self.defaults['weight_decay'],
			'amsgrad': self.defaults['amsgrad'],
		}
	

	def _checkpoint_data(self):
		data = {'settings': super().settings()}
		data['state_dict'] = self.state_dict()
		return data
	

	def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
		settings = data['settings']
		if self._is_prepared:
			current = self.settings()
			if current != settings:
				error_keys = [key for key in settings if settings.get(key) != current.get(key)]
				errors = ', '.join([f'{key}: {settings[key]} != {current[key]}' for key in error_keys])
				raise ValueError(f'settings do not match: {errors}')
			
		self.defaults['lr'] = settings['lr']
		self.defaults['betas'] = (settings['beta1'], settings['beta2'])
		self.defaults['eps'] = settings['eps']
		self.defaults['weight_decay'] = settings['weight_decay']
		self.defaults['amsgrad'] = settings['amsgrad']

		if 'state_dict' in data:
			if not self._is_prepared:
				self.prepare(None)
			self.load_state_dict(data['state_dict'])










