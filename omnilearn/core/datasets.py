from .imports import *
from .abstract import AbstractDataset, AbstractGadget
from .planning import DefaultPlanner
from omniply.apps.training import Dataset as _DatasetBase

class DatasetBase(_DatasetBase, ToolKit, AbstractDataset):
    _Planner = DefaultPlanner
    def __init__(self, gap: dict[str, str] = None, **kwargs):
        super().__init__(gap=gap, **kwargs)


    def batch(self, batch_size: Optional[int] = None, *gadgets: AbstractGadget, 
              show_pbar=False, shuffle=True, **settings: Any) -> Batch:
        return next(self.iterate(batch_size, *gadgets, show_pbar=show_pbar, shuffle=shuffle, **settings))


    def iterate(self, batch_size: Optional[int] = None, *gadgets: AbstractGadget, 
                show_pbar: bool = True, count_samples: bool = False, max_epochs: int = 1,
                shuffle: bool = None, hard_budget: bool = True, drop_last: bool = False,
                seed: Optional[int] = None, **settings: Any) -> Iterator[Batch]:
        if batch_size is None:
            batch_size = self.suggest_batch_size()
        if shuffle is None:
            shuffle = seed is not None
        
        planner = self._Planner(dataset_size=self.size, max_epochs=max_epochs, shuffle=shuffle, hard_budget=hard_budget, drop_last=drop_last, seed=seed, **settings)
   
        pbar = None
        if show_pbar:
            import tqdm
            pbar_type = tqdm.notebook.tqdm if where_am_i() == 'jupyter' else tqdm.tqdm
            total = self.size if count_samples else (self.size + batch_size - 1) // batch_size
            pbar = pbar_type(total=total, unit='x' if count_samples else 'it')
        
        try:
            while True:
                info = planner.step(batch_size)
                batch = self._Batch(info, planner=planner, allow_draw=False)
                batch.include(self).extend(gadgets)

                yield batch
                
                if show_pbar:
                    pbar.update(batch.size if count_samples else 1)
        except planner._BudgetExceeded:
            pass

