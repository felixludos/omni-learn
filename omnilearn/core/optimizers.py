from .imports import *
from .abstract import AbstractOptimizer, AbstractModel, AbstractBatch


class OptimizerBase(AbstractOptimizer):
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
	

	def setup(self, model: AbstractModel, *, device: Optional[str] = None) -> Self:
		return self



from torch import optim as O



class PytorchOptimizer(OptimizerBase, O.Optimizer):
	def __init__(self, params=None, **kwargs):
		if params is None:
			params = [torch.zeros(0)]
		super().__init__(params=params, **kwargs)
		self.param_groups.clear()


	def add_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> None:
		param_groups = list(parameters)
		if len(param_groups) == 0:
			raise ValueError("optimizer got an empty parameter list")
		if not isinstance(param_groups[0], dict):
			param_groups = [{'params': param_groups}]

		for param_group in param_groups:
			self.add_param_group(param_group)


	def setup(self, model: AbstractModel, *, device: Optional[str] = None) -> Self:
		self.add_parameters(model.parameters())
		return super().setup(model, device=device)
		

	def step(self, batch: AbstractBatch) -> AbstractBatch:
		objective = batch[self.objective]
		if self._maximize:
			objective = -objective
		objective.backward()
		super(AbstractOptimizer, self).step()
		self.zero_grad()
		return batch




class SGD(PytorchOptimizer, O.SGD):
	pass


class Adam(PytorchOptimizer, O.Adam):
	pass
