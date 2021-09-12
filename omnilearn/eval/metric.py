

from .. import util

# TODO

class MetricBase:
	def compute(self, *args, **kwargs):
		out = self._compute(*args, **kwargs)
		if isinstance(out, dict):
			scores = {score: out.get(score, None) for score in self.get_scores()}
			results = {result: out.get(result, None) for result in self.get_results()}
			out = scores, results
		return out


	def _compute(self, *args, **kwargs):
		raise NotImplementedError


	def get_scores(self):
		return []


	def get_results(self):
		return []



class Metric(util.Configurable, MetricBase):
	pass



# class EvaluationManager(util.StatsClient):
# 	pass


