from .imports import *
from .abstract import AbstractTrainer, AbstractEvent, AbstractReporter, AbstractPlanner, AbstractBatch



class ReporterBase(ToolKit, AbstractReporter):
    def setup(self, trainer, planner, batch_size):
        return self
    
    def step(self, batch):
        pass

    def end(self, last_batch=None):
        pass



class Pbar_Reporter(ReporterBase):
    def __init__(self, *, show_pbar: bool = True, count_samples: bool = True, 
                 target_ema_scale=1000, **kwargs):
        super().__init__(**kwargs)
        self._show_pbar = show_pbar
        self._count_samples = count_samples
        self._pbar = None
        self._objective_key = None
        self._target_ema_scale = target_ema_scale
        self._ema_beta = None


    _max_beta = 0.01
    def setup(self, trainer: AbstractTrainer, planner: AbstractPlanner, batch_size: int) -> Self:
        total_iterations = planner.expected_iterations(batch_size)

        # self.gauge_apply({'objective': trainer.optimizer.objective})
        self._objective_key = trainer.optimizer.objective
        self._objective_ema = None
        self._ema_beta = min(self._target_ema_scale / total_iterations, self._max_beta)

        total = planner.expected_samples(batch_size) if self._count_samples else total_iterations

        if self._show_pbar:
            import tqdm
            pbar_type = tqdm.notebook.tqdm if where_am_i() == 'jupyter' else tqdm.tqdm
            self._pbar = pbar_type(total=total, unit='x' if self._count_samples else 'it')

        return super().setup(trainer, planner, batch_size)


    def step(self, batch: AbstractBatch) -> None:
        if self._pbar is not None:
            desc = batch.grab('pbar_desc', None)
            if self._objective_key is not None:
                val = batch[self._objective_key]
                if not isinstance(val, (float, int)):
                    val = val.item()
                self._objective_ema = val if self._objective_ema is None \
                    else self._ema_beta * val + (1 - self._ema_beta) * self._objective_ema
            if desc is None:
                desc = self._default_pbar_desc(self._objective_ema)
            self._pbar.set_description(desc, refresh=False)
            self._pbar.update(batch.size if self._count_samples else 1)


    def _default_pbar_desc(self, objective: Union[float, torch.Tensor]) -> str:
        if objective >= 1000 or objective <= 0.001:
            val = f'{objective:.3g}'
        elif objective >= 100:
            val = f'{int(objective*10)/10:.1f}'
        elif objective >= 10:
            val = f'{int(objective*100)/100:.2f}'
        else:
            val = f'{int(objective*1000)/1000:.3f}'
        return f'{self._objective_key} = {val}'


    def end(self, last_batch: AbstractBatch = None) -> None:
        if self._pbar is not None:
            self._pbar.close()



class WandB_Reporter(ReporterBase):
    def __init__(self, *, indicators: Dict[str, int] = None, project_name: Optional[str] = None, 
                 project_settings: Optional[Dict[str, Any]] = None, **kwargs):
        if indicators is None:
            indicators = {}
        super().__init__(**kwargs)
        self._project_name = project_name
        self._project_settings = project_settings
        self._indicators = indicators


    def setup(self, trainer: AbstractTrainer, planner: AbstractPlanner, batch_size: int) -> Self:
        project_name = self._project_name or trainer.name
        project_settings = self._project_settings or trainer.settings

        self._indicators.update(trainer.all_indicators())

        import wandb
        wandb.init(project=project_name, config=project_settings)
        return super().setup(trainer, planner, batch_size)


    def step(self, batch: AbstractBatch) -> None:
        import wandb
        itr = batch['iteration']
        wandb.log({key: batch[key] 
                   for key, freq in self._indicators.items() 
                   if freq > 0 and itr % freq == 0})


    def end(self, last_batch: AbstractBatch = None) -> None:
        import wandb
        wandb.finish()




