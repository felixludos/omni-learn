from .imports import *


class Meter:
	def __init__(self, alpha: float = None, window_size: float = None, **kwargs):
		assert alpha is None or window_size is None, 'cannot specify both alpha and window_size'
		if window_size is not None:
			alpha = self.window_size_to_alpha(window_size)
		super().__init__(**kwargs)
		self._alpha = alpha
		self.reset()

	@staticmethod
	def window_size_to_alpha(window_size: float) -> float:
		assert window_size >= 1, f'window_size {window_size} must be >= 1'
		return 2 / (window_size + 1)

	@property
	def alpha(self):
		return self._alpha

	def reset(self):
		self.last = None
		self.avg = None
		self.sum = 0.
		self.count = 0
		self.max = None
		self.min = None
		self.smooth = None
		self.var = None
		self.std = None
		self.S = 0.
		self.timestamp = None

	def mete(self, val: float, *, n=1):
		if isinstance(val, torch.Tensor):
			val = val.detach()
			val = val.item() if val.numel() == 1 else val.numpy()
		self.timestamp = time.time()
		self.last = val
		self.sum += val * n
		prev_count = self.count
		self.count += n
		self.avg = self.sum / self.count
		delta = val - self.avg
		self.S += delta ** 2 * n * prev_count / self.count
		self.max = val if self.max is None else np.maximum(self.max, val)
		self.min = val if self.min is None else np.minimum(self.min, val)
		self.var = self.S / self.count
		self.std = np.sqrt(self.var)
		alpha = self.alpha
		if alpha is not None:
			self.smooth = val if self.smooth is None else (self.smooth * (1 - alpha) + val * alpha)
		return val

	@property
	def current(self):
		return self.last if self.smooth is None else self.smooth

	@property
	def estimate(self):
		return self.avg if self.smooth is None else self.smooth

	def __len__(self):
		return self.count

	def __float__(self):
		return self.current


# def join(self, other):
# 	'''other is another average meter, assumed to be more uptodate (for val)
# 		this does not mutate other'''
# 	raise NotImplementedError # an implementation can be found in `old/`: `AverageMeter`


_ln2 = np.log(2)


class DynamicMeter(Meter):
	def __init__(self, alpha: float = None, *, infer_alpha: bool = None, max_window_size: float = 1000,
				 min_window_size: float = 10, wait_steps: int = 5, target_halflife: float = 5, **kwargs):
		'''
		target_halflife: the time in sec
		'''
		if infer_alpha is None:
			infer_alpha = alpha is None
		super().__init__(alpha=alpha, **kwargs)
		assert max_window_size >= min_window_size, f'max_window_size {max_window_size} must be >= min_window_size {min_window_size}'
		self._max_alpha = self.window_size_to_alpha(min_window_size)
		self._min_alpha = self.window_size_to_alpha(max_window_size)
		self._wait_steps = wait_steps
		self._target_halflife = target_halflife
		self._alpha_estimator = Meter(window_size=self._wait_steps) if infer_alpha else None
		self._hanging_tick = None

	@property
	def alpha(self):
		if self._alpha_estimator is not None:
			estimate = self._alpha_estimator.estimate
			if estimate is not None:
				return min(max(self._min_alpha, self._alpha_estimator.estimate), self._max_alpha)
		return self._alpha

	def mete(self, val: float, *, n=1):
		val = super().mete(val, n=n)
		if self._alpha_estimator is not None:
			if self._hanging_tick is not None:
				dt = self.timestamp - self._hanging_tick
				alpha = 1 - np.exp(-_ln2 / (self._target_halflife / dt))
				self._alpha_estimator.mete(alpha)
				self._hanging_tick = None
			if self._wait_steps is None or self.count >= self._wait_steps:
				self._hanging_tick = self.timestamp
		return val


class IntervalMeter(DynamicMeter):
	def __init__(self, alpha: float = None, window_size: float = None, **kwargs):
		super().__init__(alpha=alpha, window_size=window_size, **kwargs)
		self._last_timestamp = None

	def mete(self, *, n=1):
		now = time.time()
		out = None
		if self._last_timestamp is not None:
			out = super().mete(now - self._last_timestamp, n=n)
		self._last_timestamp = now
		return out



