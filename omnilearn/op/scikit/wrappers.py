import torch
from sklearn import base
from ..framework import Learner, FunctionBase
from ...eval import EvaluatorBase

from ... import util

class ScikitWrapper:

	def _format_scikit_arg(self, data):
		if data is not None and isinstance(data, torch.Tensor):
			data = data.cpu().numpy()
		return data



class Estimator(util.Cached, Learner, ScikitWrapper, base.BaseEstimator):
	
	def fit(self, data, targets=None):
		return self._fit(data, targets=targets)
	
	
	def _fit(self, data, targets=None):
		data = self._format_scikit_arg(data)
		targets = self._format_scikit_arg(targets)
		return super(Estimator, self).fit(data, targets=targets)
	
	
	def predict(self, data):
		return self._predict(data)
	
	
	def _predict(self, data):
		data = self._format_scikit_arg(data)
		return super(Estimator, self).predict(data)
	
	
	def predict_probs(self, data):
		return self._predict_probs(data)
	
	
	def _predict_probs(self, data):
		data = self._format_scikit_arg(data)
		return super(Estimator, self).predict_proba(data)
	
	
	def predict_score(self, data):
		return self._predict_score(data)
	
	
	def _predict_score(self, data):
		return super(Estimator, self).predict_scores(data)
	
	
	def cache_results(self, data):
		return self.ResultContext(self, data)
	
	
	class ResultContext:
		def __init__(self, estimator, data):
			self.estimator = estimator
			self.data = data
			self.results = None
		
		def predict(self):
			pass
		
		def __enter__(self):
			self.results = {}
			
		def __exit__(self, exc_type, exc_val, exc_tb):
			pass
	

class Supervised(Estimator):
	pass
	


class Classifier(Supervised, base.ClassifierMixin):
	pass



class Regressor(Supervised, base.RegressorMixin):
	pass



class Clustering(Estimator, base.ClusterMixin):
	pass



class Transformer(Estimator, base.TransformerMixin):
	def transform(self, data):
		raise NotImplementedError
	
	def fit_transform(self, data):
		raise NotImplementedError





class Metric(ScikitWrapper):
	_name = None
	_result_name = None
	_fn = None
	_arg_reqs = {}
	
	
	def _build_args(self, model, args, labels=None):
		for argname, req in _args_reqs.items():
			if argname not in args:
				args[argname] = model.demand_result(req)
		return args
	
	
	@classmethod
	def get_name(cls):
		return cls._name
	
	
	def _dispatch(self, **kwargs):
		return self._fn(**kwargs)
	
	
	def _compute(self, estimator, dataset, **kwargs):
		
		kwargs = self._build_args(estimator, kwargs, dataset)
		
		
		
		pass
	
	




