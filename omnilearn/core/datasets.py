from omnibelt import closest_factors, prime_factors
from omniply.apps.training import Dataset as _DatasetBase
from omniply.apps.viz import Context as VizContext

from .imports import *
from .abstract import AbstractDataset, AbstractGadget, AbstractFileDataset
from .planning import DefaultPlanner



class VizBatch(Batch, VizContext):
	def __init__(self, *args, start_recording: bool = True, **kwargs):
		super().__init__(*args, **kwargs)
		if start_recording:
			self.record()


class DatasetBase(AbstractDataset):
	_Batch = Batch
	_Planner = DefaultPlanner

	def load(self, *, device: Optional[str] = None) -> Self:
		return self


	def suggest_batch_size(self, *, prefer_power_of_two: bool = True, 
							target_iterations: Optional[int] = 100, 
							target_batch_size: Optional[int] = None) -> int:
		return next(suggest_batch_sizes(self.size, prefer_power_of_two=prefer_power_of_two, 
									   target_iterations=target_iterations, target_batch_size=target_batch_size))


	def batch(self, batch_size: Optional[int] = None, *gadgets: AbstractGadget, 
			  shuffle=True, **settings: Any) -> Batch:
		return next(self.iterate(batch_size, *gadgets, show_pbar=False, shuffle=shuffle, **settings))


	def iterate(self, batch_size: Optional[int] = None, *gadgets: AbstractGadget, 
				show_pbar: bool = True, count_samples: bool = True, pbar_desc=None,
				max_epochs: int = 1, hard_budget: bool = True, drop_last: bool = False,
				shuffle: bool = None, seed: Optional[int] = None, 
				**settings: Any) -> Iterator[Batch]:
		if batch_size is None:
			batch_size = self.suggest_batch_size()
		if shuffle is None:
			shuffle = seed is not None
		
		planner = self._Planner(dataset_size=self.size, max_epochs=max_epochs, shuffle=shuffle, hard_budget=hard_budget, drop_last=drop_last, seed=seed, **settings)
   
		pbar = None
		if show_pbar:
			import tqdm
			pbar_type = tqdm.notebook.tqdm if where_am_i() == 'jupyter' else tqdm.tqdm
			total = self.size if count_samples else planner.expected_iterations(batch_size)
			if pbar_desc is None:
				pbar_desc = self.name
			pbar = pbar_type(total=total, unit='x' if count_samples else 'it', desc=pbar_desc)
		
		for info in planner.generate(batch_size):
			batch = self._Batch(info, planner=planner, allow_draw=False)

			yield batch.include(self).extend(gadgets)

			if show_pbar:
				pbar.update(batch.size if count_samples else 1)



class FileDatasetBase(DatasetBase, AbstractFileDataset):
	@property
	def dataroot(self) -> Optional[Path]:
		return None



def suggest_batch_sizes(dataset_size: int, *, 
						prefer_power_of_two: bool = True,
						target_iterations: Optional[int] = 100, 
						target_batch_size: Optional[int] = None) -> Iterator[int]:
	'''
	yields perfectly divisible batch sizes in order of closeness to target_batch_size 
	(or s.t. there are target_iterations if not provided)

	if prefer_power_of_two is True, it will first yield the best power of two
	'''
	if dataset_size is None:
		yield 32
		return

	assert target_batch_size is not None or target_iterations is not None, 'either target_batch_size or target_iterations must be provided'
	if target_batch_size is None:
		target_batch_size = max(int(math.sqrt(dataset_size)), dataset_size // target_iterations)
	assert 0 < target_batch_size <= dataset_size, 'target_batch_size must be in (0, dataset_size]' 

	factors = Counter(prime_factors(dataset_size))
	if prefer_power_of_two and 2 in factors:
		yield min((2 ** i for i in range(1,factors[2] + 1)), key=lambda x: abs(x - target_batch_size))

	yield from closest_factors(factors, Counter(prime_factors(target_batch_size)))


