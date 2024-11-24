from .imports import *
from .common import Machine, Event

from ..training import CheckpointableTrainer, Pbar_Reporter, Checkpointer as _Checkpointer, DefaultPlanner
from ..training import WandB_Monitor as _WandB_Monitor



class Planner(Machine, DefaultPlanner):
	pass



class WandB_Monitor(Event, _WandB_Monitor):
	def __init__(self, *, freqs: Dict[str, int] = None, project_name: str = None, use_wandb: bool = None, **kwargs):
		super().__init__(freqs=freqs, project_name=project_name, use_wandb=use_wandb, **kwargs)



class Reporter(Machine, Pbar_Reporter):
	@fig.config_aliases(print_metrics='log')
	def __init__(self, print_metrics: Iterable[str] = None, **kwargs):
		super().__init__(print_metrics=print_metrics, **kwargs)



class Checkpointer(Machine, _Checkpointer):
	def __init__(self, saveroot: Path = 'checkpoints', *, freq: int = None, skip_0: bool = True, **kwargs):
		if saveroot is None:
			print(f'WARNING: No saveroot provided, so no checkpoints will be saved')
			freq = None
		else:
			saveroot = Path(saveroot)
		super().__init__(saveroot=saveroot, freq=freq, skip_0=skip_0, **kwargs)



class Trainer(Configurable, CheckpointableTrainer):
	_Planner = Planner
	_Reporter = Reporter
	def __init__(self, model: AbstractModel, optimizer: AbstractOptimizer, *, reporter: AbstractEvent = None,
				 env: Dict[str, AbstractMachine] = None, events: Dict[str, AbstractEvent] = None,
				 planner: AbstractPlanner = None, budget: Union[int, Dict[str, int]] = None, batch_size: int = None,
				 device: str = None, **kwargs):
		if isinstance(budget, int):
			budget = {'max_iterations': budget}
		super().__init__(model=model, optimizer=optimizer, reporter=reporter, planner=planner, env=env, events=events,
						 batch_size=batch_size, device=device, **kwargs)
		if budget is not None:
			self._planner.budget(**budget)



