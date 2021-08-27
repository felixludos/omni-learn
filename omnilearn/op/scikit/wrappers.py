import torch
from sklearn import base, metrics
from ..framework import Learner, FunctionBase, Evaluatable
from ...eval import EvaluatorBase

from ... import util

class ScikitWrapper:
	def _format_scikit_arg(self, data):
		if data is not None and isinstance(data, torch.Tensor):
			data = data.cpu().numpy()
		return data



class ScikitEstimatorInfo(ScikitWrapper, util.TensorDict):
	def __init__(self, estimator, dataset):
		self.__dict__['_estimator'] = estimator
		self.__dict__['dataset'] = dataset
		self._process_dataset()


	def _process_dataset(self):
		self.observations = self.dataset.get_observations()
		self.labels = self.dataset.get_labels()
		self.__dict__['_estimator_observations'] = self._format_scikit_arg(self.observations)
		self.__dict__['_estimator_labels'] = self._format_scikit_arg(self.labels)


	def get_observations(self):
		return self._estimator_observations


	def get_labels(self):
		return self._estimator_labels


	def get_result(self, key):
		if key not in self:
			out = self._estimator.make_prediction(key, self._estimator_observations)
			self[key] = self._format_scikit_arg(out)
		return self[key]



class ScikitEstimator(Learner, EvaluatorBase, ScikitWrapper):

	def get_params(self):
		try:
			return super(Learner, self).get_params()
		except AttributeError:
			return self.state_dict()


	def set_params(self, **params):
		try:
			return super(Learner, self).set_params(**params)
		except AttributeError:
			return self.load_state_dict(params)


	def _fit(self, dataset, out=None):
		if out is None:
			out = util.ScikitEstimatorInfo(self, dataset)
		super(Learner, self).fit(out._estimator_observations, targets=out._estimator_labels)
		return out


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


	def make_prediction(self, name, data):
		methods = {'pred': self.predict,
		           'scores': self.predict_score,
		           'probs': self.predict_probs}
		method = methods.get(name, None)
		if method is not None:
			return method(data)


	def _compute(self, run):
		if not self.is_fit():
			dataset = run.get_dataset()
			out = self.fit(dataset)
		return self.evaluate(run, out=out)


	def evaluate(self, info, out=None, **kwargs):
		dataset = info.get_dataset()
		if out is None:
			out = util.ScikitEstimatorInfo(self, dataset)
		return self._evaluate(out, **kwargs)


	def _evaluate(self, info, **kwargs):
		return info



class Supervised(ScikitEstimator):
	pass
	


class Classifier(Supervised, base.ClassifierMixin):

	def _evaluate(self, info, **kwargs):
		info = super()._evaluate(info, **kwargs)

		labels, pred = info.get_labels(), info.get_result('pred')

		report = metrics.classification_report(labels, pred, target_names=info.dataset.get_label_names(),
		                                       output_dict=True)
		confusion = metrics.confusion_matrix(labels, pred)

		precision, recall, fscore, support = metrics.precision_recall_fscore_support(labels, pred)

		scores = info.get_result('scores')
		auc = metrics.roc_auc_score(labels, scores)

		roc = None
		if labels.max() == 1:
			scores = scores.reshape(-1)
			roc = metrics.roc_curve(labels, scores)

		info.update({
			'roc-auc': auc.mean(),
			'f1': fscore.mean(),
			'precision': precision.mean(),
			'recall': recall.mean(),

			'worst-roc-auc': auc.min(),
			'worst-f1': fscore.min(),
			'worst-precision': precision.min(),
			'worst-recall': recall.min(),

			'full-roc-auc': auc,
			'full-f1': fscore,
			'full-recall': recall,
			'full-precision': precision,
			'full-support': support,
			'report': report,
			'confusion': confusion,
		})
		if roc is not None:
			info['roc-curve'] = roc
		return info

	def get_scores(self):
		return ['roc-auc', 'f1', 'precision', 'recall',
			'worst-roc-auc', 'worst-f1', 'worst-precision', 'worst-recall',
			*super().get_scores()]

	def get_results(self):
		return ['report', 'confusion',
		        'full-roc-auc', 'roc-curve',
		        'full-f1', 'full-recall', 'full-precision', 'full-support',
		        *super().get_results()]



class Regressor(Supervised, base.RegressorMixin):

	def _evaluate(self, info, **kwargs):
		info = super()._evaluate(info, **kwargs)

		labels, pred = info.get_labels(), info.get_result('pred')

		mse = metrics.mean_squared_error(labels, pred)
		mxe = metrics.max_error(labels, pred)
		mae = metrics.mean_absolute_error(labels, pred)
		medae = metrics.median_absolute_error(labels, pred)
		r2 = metrics.r2_score(labels, pred)

		info.update({
			'mse': mse,
			'mxe': mxe,
			'mae': mae,
			'medae': medae,
			'r2-score': r2,
		})

		return info

	def get_scores(self):
		return ['mse', 'mxe', 'mae', 'medae', 'r2-score', *super().get_scores()]



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
	
	




