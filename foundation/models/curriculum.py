
import omnifig as fig

class Signal:
	def __init__(self, name=None):

		self.name = name

		self.meter = None

	def add_to_stats(self, stats):
		if self.name is None:
			raise Exception(f'signal {self} doesnt have a name for stats')
		stats.new(self.name)
		self.meter = stats

	def step(self):
		val = self._step()

		return val

	def _step(self):
		pass


