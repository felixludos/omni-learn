from .imports import *

from omniply.apps.training.abstract import AbstractDataset as AbstractDatasetBase, AbstractPlanner, AbstractBatch, AbstractTrainer as AbstractTrainerBase


class AbstractDataset(AbstractDatasetBase):
    @property
    def name(self) -> str:
        raise NotImplementedError
    

    def load(self, *, device: Optional[str] = None) -> Self:
        raise NotImplementedError


    @property
    def dataroot(self) -> Path:
        raise NotImplementedError


    def suggest_batch_size(self) -> int:
        pass


    def __iter__(self) -> Iterator[AbstractBatch]:
        return self.iterate()



class AbstractModel(AbstractGadget):
    @property
    def name(self) -> str:
        raise NotImplementedError
    

    def prepare(self, dataset: AbstractDataset, *, device: Optional[str] = None) -> Self:
        raise NotImplementedError



class AbstractOptimizer(AbstractGadget):    
    def setup(self, model: AbstractModel, *, device: Optional[str] = None) -> Self:
        raise NotImplementedError


    def step(self, batch: AbstractBatch) -> None:
        raise NotImplementedError
    

    @property
    def objective(self) -> str:
        '''key of the overall objective being optimized (default is "loss")'''
        raise NotImplementedError
    

    @property
    def objective_direction(self) -> int:
        '''direction of the objective (1 for maximization, -1 for minimization)'''
        raise NotImplementedError



class AbstractTrainer(AbstractTrainerBase):
    @property
    def name(self) -> str:
        raise NotImplementedError
    

    @property
    def model(self) -> AbstractModel:
        raise NotImplementedError
    

    @property
    def optimizer(self) -> AbstractOptimizer:
        raise NotImplementedError
    

    @property
    def dataset(self) -> AbstractDataset:
        raise NotImplementedError



class AbstractReporter(AbstractGadget):
    def setup(self, trainer: AbstractTrainer, planner: AbstractPlanner, batch_size: int) -> Self:
        raise NotImplementedError


    def step(self, batch: AbstractBatch) -> None:
        raise NotImplementedError
    

    def end(self, last_batch: AbstractBatch = None) -> None:
        raise NotImplementedError



class AbstractEvent(AbstractGadget):
    def setup(self, trainer: AbstractTrainer, *, device: Optional[str] = None) -> Self:
        raise NotImplementedError


    def step(self, batch: AbstractBatch) -> None:
        raise NotImplementedError
    

    def end(self, last_batch: AbstractBatch = None) -> None:
        raise NotImplementedError

