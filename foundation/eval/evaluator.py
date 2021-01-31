

from .. import util

# TODO

class Evaluator(util.Configurable):
	
	def compute(self, *args, **kwargs):
		out = self._compute(*args, **kwargs)
		if isinstance(out, dict):
			scores = {score:out.get(score, None) for score in self.get_scores()}
			results = {result:data for result, data in out.items() if result not in scores}
			out = scores, results
		return out
	
	def _compute(self, *args, **kwargs):
		raise NotImplementedError
	
	def get_scores(self):
		return []
	
	def get_results(self):
		return []
	
	pass



class EvaluationManager(util.StatsClient):
	
	
	
	pass


