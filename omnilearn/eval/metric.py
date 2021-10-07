

from .. import util

# TODO

class MetricBase:
	def compute(self, *args, **kwargs):
		out = self._compute(*args, **kwargs)
		return self._process_results(out)


	def _breakdown_val(self, val):
		try:
			return val.mean().item()
		except AttributeError:
			try:
				return val.item()
			except AttributeError:
				return val


	def _process_results(self, out):
		if isinstance(out, dict):
			scores = {score: out.get(score, None) for score in self.get_scores() if out.get(score, None) is not None}
			results = {result: out.get(result, None) for result in self.get_results()
			           if out.get(result, None) is not None}
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


