import numpy as np
import omnibelt as belt
from omnibelt import unspecified_argument
import omnifig as fig
import torch
from torch import nn
from sklearn import base, metrics
from ..framework import Learnable, FunctionBase, Evaluatable, Recordable, Function
from ...eval import MetricBase

from ... import util

class ScikitWrapper(belt.InitWall):
	def __init__(self, *args, _multi_inits=None, _req_args=None,
	             _req_kwargs=None, **kwargs):
		if isinstance(self, nn.Module):
			_multi_inits = [None, nn.Module]
			super().__init__(*args, _multi_inits=_multi_inits,
			                 _req_args={}, _req_kwargs={nn.Module:kwargs},
			                 **kwargs)
		else:
			super().__init__(*args, _multi_inits=_multi_inits,
			                 _req_args=_req_args, _req_kwargs=_req_kwargs,
			                 **kwargs)
	
	
	def _format_scikit_arg(self, data):
		if data is not None and isinstance(data, torch.Tensor):
			data = data.cpu().numpy()
		return data

	def _format_scikit_output(self, out):
		if out is not None and isinstance(out, np.ndarray):
			out = torch.from_numpy(out)
		return out


class ScikitEstimatorInfo(ScikitWrapper, util.TensorDict):
	def __init__(self, estimator, dataset):
		self.__dict__['_estimator'] = estimator
		self.__dict__['dataset'] = dataset
		self._process_dataset()


	def _process_dataset(self):
		# self.observations = self.dataset.get_observations()
		# self.labels = self.dataset.get_labels()
		self.__dict__['_estimator_observations'] = self._format_scikit_arg(self.dataset.get_observations())
		self.__dict__['_estimator_labels'] = self._format_scikit_arg(self.dataset.get_labels())


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
		super(Learnable, self).fit(out.get_observations(), out.get_labels())
		return out


	def predict(self, data):
		return self._predict(data)
	
	
	def _predict(self, data):
		data = self._format_scikit_arg(data)
		return self._format_scikit_output(super(ScikitWrapper, self).predict(data))
	
	
	def predict_probs(self, data):
		return self._predict_probs(data)
	
	
	def _predict_probs(self, data):
		data = self._format_scikit_arg(data)
		return self._format_scikit_output(super(ScikitWrapper, self).predict_proba(data))
	
	
	def predict_score(self, data):
		return self._predict_score(data)
	
	
	def _predict_score(self, data):
		return self._format_scikit_output(super(ScikitWrapper, self).predict_scores(data))


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


	def evaluate(self, dataset=None, info=None, out=None, **kwargs):
		if dataset is None:
			dataset = info.get_dataset()
		if out is None:
			out = ScikitEstimatorInfo(self, dataset)
		return self._evaluate(out, **kwargs)


	def _evaluate(self, info, **kwargs):
		return info



# class ModelBuilder(util.Builder):
# 	def build(self, dataset):
# 		return None


class ScikitEstimator(Recordable, ScikitEstimatorBase):
	pass



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

		auc = None
		roc = None
		if labels.max() == 1:
			scores = info.get_result('scores')
			auc = metrics.roc_auc_score(labels, scores)

			scores = scores.reshape(-1)
			roc = metrics.roc_curve(labels, scores)

		info.update({
			'f1': fscore.mean(),
			'precision': precision.mean(),
			'recall': recall.mean(),

			'worst-f1': fscore.min(),
			'worst-precision': precision.min(),
			'worst-recall': recall.min(),

			'full-f1': fscore,
			'full-recall': recall,
			'full-precision': precision,
			'full-support': support,
			'report': report,
			'confusion': confusion,
		})
		if roc is not None:
			info['roc-curve'] = roc
		if auc is not None:
			info['roc-auc'] = auc.mean()
			info['worst-roc-auc'] = auc.min()
			info['full-roc-auc'] = auc
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




class SingleLabelEstimator(ScikitEstimator): # dout must be 1 (dof)
	pass



# @fig.AutoModifier('discretized')
class Discretized(ScikitEstimator): # must be a boundd space
	pass



# @fig.AutoModifier('continuized')
class Continuized(ScikitEstimator): # must be a categorical space
	pass



class ParallelEstimator(ScikitEstimatorBase):
	def __init__(self, estimators, din=None, dout=None, **kwargs):

		super().__init__(**kwargs)
		self.estimators = estimators


	def include_estimators(self, *estimators):
		self.estimators.extend(estimators)


	def _process_inputs(self, key, *ins):
		return len(self.estimators)*[ins]


	def _process_outputs(self, key, outs):
		if 'predict' in key:
			return torch.stack(outs, -1)
		if 'evaluate' in key:
			return outs
		return util.pytorch_collate(outs)


	def _dispatch(self, key, *ins, **kwargs):
		ins = self._process_inputs(key, *ins)
		outs = [getattr(estimator, key)(*inp) for estimator, inp in zip(self.estimators, ins)]
		return self._process_outputs(key, outs)


	def _fit(self, dataset, out=None, **kwargs):
		return self._dispatch('_fit', dataset, **kwargs)


	def _predict(self, data, **kwargs):
		return self._dispatch('_predict', data, **kwargs)


	def evaluate(self, dataset=None, info=None, **kwargs):
		if dataset is None:
			dataset = info.get_dataset()
		return self._dispatch('evaluate', dataset, **kwargs)



@fig.Component('joint-estimator')
class JointEstimator(ScikitEstimator, ParallelEstimator): # collection of single dim estimators (can be different spaces)
	def __init__(self, A, estimators=unspecified_argument, **kwargs):
		if estimators is unspecified_argument:
			estimators = A.pull('estimators', [])
		super().__init__(A, estimators=estimators, **kwargs)


	def _process_inputs(self, key, dataset):
		if 'predict' in key:
			return [(dataset,) for _ in self.estimators]
		datasets = [(dataset.duplicate(),) for _ in self.estimators]
		for idx, (ds,) in enumerate(datasets):
			ds.register_wrapper('single-label', kwargs={'idx': idx})
		return datasets



class MultiEstimator(SingleLabelEstimator, ParallelEstimator): # all estimators must be of the same kind (out space)
	def __init__(self, A, raw_pred=None, **kwargs):

		if raw_pred is None:
			raw_pred = A.pull('raw_pred', False)

		super().__init__(A, **kwargs)
		self._raw_pred = raw_pred


	def toggle_raw_pred(self, val=None):
		if val is None:
			val = not self._raw_pred
		self._raw_pred = val


	def __getattribute__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return self.estimators[0].__getattribute__(item)


	def _merge_outs(self, outs, **kwargs):
		raise NotImplementedError


	def _process_outputs(self, key, outs, **kwargs):
		if 'predict' in key:
			return outs if self._raw_pred else self._merge_outs(outs, **kwargs)
		return util.pytorch_collate(outs)



class Periodized(MultiEstimator): # create a copy of the estimator for the sin component

	def _process_inputs(self, key, *ins):
		if 'fit' in key:
			infos = [ScikitEstimatorInfo(est, ins[0]) for est in self.estimators]
			infos[0]._estimator_labels = np.cos(infos[0]._estimator_labels)
			infos[1]._estimator_labels = np.sin(infos[1]._estimator_labels)
			return [(ins[0], info) for info in infos]
		return super()._process_inputs(key, *ins)

	def _merge_outs(self, outs, **kwargs):
		cos, sin = outs
		return torch.atan2(sin, cos)



@fig.Modifier('periodized', expects_config=False)
def _make_periodized(component):
	def _periodized(config):
		return Periodized(config, estimators=[component(config), component(config)])
	return _periodized






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
	
	




