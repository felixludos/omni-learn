import omnibelt as belt
from omnibelt import unspecified_argument
import omnifig as fig
import torch
from sklearn import base, metrics
from ..framework import Learnable, FunctionBase, Evaluatable, Recordable, Function
from ...eval import MetricBase

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



class ScikitEstimatorBase(Learnable, MetricBase, ScikitWrapper):

	def register_out_space(self, space):
		self._outspace = space


	def get_params(self):
		try:
			return super(Learnable, self).get_params()
		except AttributeError:
			return self.state_dict()


	def set_params(self, **params):
		try:
			return super(Learnable, self).set_params(**params)
		except AttributeError:
			return self.load_state_dict(params)


	def _fit(self, dataset, out=None):
		if out is None:
			out = ScikitEstimatorInfo(self, dataset)
		super(Learnable, self).fit(out._estimator_observations, out._estimator_labels)
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


	def _compute(self, run, dataset=None):
		if dataset is None:
			dataset = run.get_dataset()
		if not self.is_fit():
			out = self.fit(dataset)
		return self.evaluate(run, out=out, dataset=dataset)


	def evaluate(self, info, out=None, dataset=None, **kwargs):
		if dataset is None:
			dataset = info.get_dataset()
		if out is None:
			out = ScikitEstimatorInfo(self, dataset)
		return self._evaluate(out, **kwargs)


	def _evaluate(self, info, **kwargs):
		return info



class ModelBuilder(util.Builder):
	def build(self, dataset):
		return None



class Supervised(ScikitEstimatorBase):
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



class Clustering(ScikitEstimatorBase, base.ClusterMixin):
	pass



class Transformer(ScikitEstimatorBase, base.TransformerMixin):
	def transform(self, data):
		raise NotImplementedError
	
	def fit_transform(self, data):
		raise NotImplementedError




class ScikitEstimator(Recordable, ScikitEstimatorBase):

	pass



class SingleLabelEstimator(ScikitEstimator): # dout must be 1 (dof)
	pass



# @fig.AutoModifier('discretized')
class Discretized(ScikitEstimator): # must be a boundd space
	pass



# @fig.AutoModifier('continuized')
class Continuized(ScikitEstimator): # must be a categorical space
	pass



@fig.Component('joint-estimator')
class JointEstimator(ScikitEstimator): # collection of single dim estimators (can be different spaces)
	def __init__(self, A, estimators=unspecified_argument, **kwargs):
		if estimators is unspecified_argument:
			estimators = A.pull('estimators', [])
		super().__init__(A, **kwargs)
		self.estimators = estimators


	def include_estimators(self, *estimators):
		self.estimators.extend(estimators)


	def _process_datasets(self, dataset):
		datasets = [dataset.duplicate() for _ in self.estimators]
		for idx, ds in enumerate(datasets):
			ds.register_wrapper('single-label', kwargs={'idx': idx})
		return datasets


	def _dispatch(self, key, dataset, **kwargs):
		datasets = self._process_datasets(dataset)
		outs = [getattr(estimator, key)(dataset) for estimator, dataset in zip(self.estimators, datasets)]
		return self._collate_outs(outs)


	def _collate_outs(self, outs):
		return util.pytorch_collate(outs)


	def _fit(self, dataset, out=None, **kwargs):
		return self._dispatch('_fit', dataset, **kwargs)


	def _evaluate(self, dataset, out=None, **kwargs):
		return self._dispatch('_evaluate', dataset, **kwargs)



class MultiEstimator(SingleLabelEstimator): # all estimators must be of the same kind (out space)
	def __init__(self, estimator_info, _is_extra=None, extra_estimators=None, **kwargs):

		if _is_extra is None:
			_is_extra = A.pull('_is_extra', False, silent=True)

		if not _is_extra:
			if extra_estimators is None:
				extra_estimators = A.pull('extra-estimators', None)
				if extra_estimators is None:
					extra_estimators = A.pull('dout', None)
					if extra_estimators is not None:
						extra_estimators -= 1
				if extra_estimators is None:
					extra_estimators = 0

		super().__init__(**kwargs)

		extras = None
		if not _is_extra and extra_estimators > 0:
			estimator_info.push('_is_extra', True, silent=True)
			extras = [estimator_info.pull_self() for _ in range(extra_estimators)]
		self.extras = extras


	def _collate_outs(self, outs):
		return util.pytorch_collate(outs)


	def _process(self, key, outs, dataset, **kwargs):
		if key == 'predict':
			return torch.cat(outs, -1)
		return self._collate_outs(outs)


	def _dispatch(self, key, dataset):
		outs = [getattr(estimator, key)(dataset) for estimator in self.estimators]
		return self._process(key, outs, dataset)



@fig.AutoModifier('periodized')
class Periodized(MultiEstimator): # create a copy of the estimator for the sin component
	def __init__(self, estimator_info, extra_estimators=None, **kwargs):
		super().__init__(estimator_info, extra_estimators=1, **kwargs)


	def _process(self, key, outs, dataset, **kwargs):
		cos, sin = outs
		return torch.atan2(sin, cos)




# old



# class Metric(ScikitWrapper):
# 	_name = None
# 	_result_name = None
# 	_fn = None
# 	_arg_reqs = {}
#
#
# 	def _build_args(self, model, args, labels=None):
# 		for argname, req in _args_reqs.items():
# 			if argname not in args:
# 				args[argname] = model.demand_result(req)
# 		return args
#
#
# 	@classmethod
# 	def get_name(cls):
# 		return cls._name
#
#
# 	def _dispatch(self, **kwargs):
# 		return self._fn(**kwargs)
#
#
# 	def _compute(self, estimator, dataset, **kwargs):
#
# 		kwargs = self._build_args(estimator, kwargs, dataset)
#
#
#
# 		pass
	
	




