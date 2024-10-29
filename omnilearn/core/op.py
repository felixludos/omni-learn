from .imports import *
import omnifig as fig

from .abstract import AbstractModel, AbstractOptimizer, AbstractEvent, AbstractElement
from .planning import DefaultPlanner
from .datasets import FileDatasetBase, Batch
from .models import MLP as MLPBase, ModelBase
from .trainers import TrainerBase
from .optimizers import Adam as AdamBase, SGD as SGDBase, OptimizerBase as Optimizer
from .events import PbarReporter

# configurable versions of the top level functions

class Machine(fig.Configurable, ToolKit):
	@fig.config_aliases(gap='app')
	def __init__(self, gap=None, **kwargs):
		super().__init__(gap=gap, **kwargs)



class Trainer(fig.Configurable, TrainerBase):
	_Planner = DefaultPlanner
	_Reporter = PbarReporter
	def __init__(self, model: AbstractModel, optimizer: AbstractOptimizer, *, 
			  reporter: AbstractEvent = None, env: Dict[str, AbstractElement] = None, 
			  budget: int = None, device: str = None, **kwargs):
		super().__init__(model=model, optimizer=optimizer, reporter=reporter, env=env, device=device, **kwargs)
		if budget is not None:
			self._planner._max_iterations = budget



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
	def compute_output(self, input):
		return self(input)
	# def forward(self, input):
	# 	return super().forward(input)



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








