from .imports import *
import omnifig as fig

from .abstract import AbstractModel, AbstractOptimizer, AbstractEvent, AbstractMachine
from .planning import DefaultPlanner as Planner
from .datasets import FileDatasetBase, Batch
from .models import MLP as MLPBase, ModelBase
from .trainers import TrainerBase
from .optimizers import Adam as AdamBase, SGD as SGDBase, OptimizerBase as Optimizer
from .events import Pbar_Reporter as Reporter

# configurable versions of the top level functions



class Machine(fig.Configurable, ToolKit, AbstractMachine):
	@fig.config_aliases(gap='app')
	def __init__(self, gap=None, **kwargs):
		super().__init__(gap=gap, **kwargs)


	def _checkpoint_data(self) -> Dict[str, Any]:
		return self.settings()


	def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
		pass


	def checkpoint(self, path: Path = None):
		data = self._checkpoint_data()
		if path is None:
			return data
		torch.save(data, path)
	

	def load_checkpoint(self, *, path = None, data = None):
		assert path is None != data is None, 'must provide exactly one of path or data'
		if data is None:
			data = torch.load(path)
		self._load_checkpoint_data(data)


	def settings(self):
		return {}



class Trainer(fig.Configurable, TrainerBase):
	_Planner = Planner
	_Reporter = Reporter
	def __init__(self, model: AbstractModel, optimizer: AbstractOptimizer, *, 
			  reporter: AbstractEvent = None, env: Dict[str, AbstractMachine] = None, 
			  budget: Union[int, Dict[str, int]] = None, batch_size: int = None,
			  device: str = None, **kwargs):
		if isinstance(budget, int):
			budget = {'max_iterations': budget}
		super().__init__(model=model, optimizer=optimizer, reporter=reporter, env=env, batch_size=batch_size, device=device, **kwargs)
		if budget is not None:
			self._planner.budget(**budget)



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
	def compute_output(self, input: torch.Tensor) -> torch.Tensor:
		return self(input)
	

	def settings(self):
		out = super()._checkpoint_data()
		out['hidden'] = self._hidden
		out['nonlin'] = self._nonlin
		out['output_nonlin'] = self._output_nonlin
		out['input_dim'] = self._input_dim
		out['output_dim'] = self._output_dim
		return out
	
	def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
		super()._load_checkpoint_data(data)
		self._hidden = data['hidden']
		self._nonlin = data['nonlin']
		self._output_nonlin = data['output_nonlin']
		self._input_dim = data['input_dim']
		self._output_dim = data['output_dim']



class SGD(Machine, SGDBase):
	def __init__(self, lr: float, momentum: float = 0., dampening: float = 0., 
				 weight_decay: float = 0., nesterov: bool = False, **kwargs):
		super().__init__(lr=lr, momentum=momentum, dampening=dampening, 
						 weight_decay=weight_decay, nesterov=nesterov, **kwargs)



class Adam(Machine, AdamBase):
	def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
				 eps: float = 1e-8, weight_decay: float = 0., amsgrad: bool = False, 
				 **kwargs):
		super().__init__(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, 
						 amsgrad=amsgrad, **kwargs)








