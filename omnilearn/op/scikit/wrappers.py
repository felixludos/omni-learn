import numpy as np
import omnibelt as belt
from omnibelt import unspecified_argument
import omnifig as fig
import torch
from torch import nn
from sklearn import base, metrics, cluster

from ..framework import Learnable, FunctionBase, Evaluatable, Recordable, Function, Computable
from ...data import MissingDataError

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
		self.__dict__['_estimator_observations'] = self._format_scikit_arg(self.dataset.get('observations'))
		try:
			targets = self.dataset.get('targets')
		except MissingDataError:
			targets = None
		else:
			targets = self._format_scikit_arg(targets)
			if len(targets.shape) == 2 and targets.shape[1] == 1:
				targets = targets.reshape(-1)
		self.__dict__['_estimator_targets'] = targets


	def get_observations(self):
		return self._estimator_observations


	def get_targets(self):
		return self._estimator_targets


	def get_result(self, key):
		if key not in self:
			out = self._estimator.make_prediction(key, self._estimator_observations)
			self[key] = self._format_scikit_arg(out)
		return self[key]



class AutomaticOutlierInfo(ScikitEstimatorInfo):
	def __init__(self, estimator, dataset, auto=None, remove_training=False):
		self.__dict__['_auto_class'] = auto
		self.__dict__['_auto_remove'] = remove_training
		super().__init__(estimator=estimator, dataset=dataset)


	def _process_dataset(self):
		# self.observations = self.dataset.get_observations()
		# self.labels = self.dataset.get_labels()
		obs = self.dataset.get('observations')
		try:
			targets = self.dataset.get('targets')
		except MissingDataError:
			targets = None
		else:
			if len(targets.shape) == 2 and targets.shape[1] == 1:
				targets = targets.view(-1)
			if self._auto_class is not None:
				if self._auto_remove and 'train' in self.dataset.get_mode():
					inds = targets != self._auto_class
					obs = obs[inds]
					targets = targets[inds]
				else:
					targets = targets == self._auto_class
			targets = self._format_scikit_arg(targets)

		self.__dict__['_estimator_observations'] = self._format_scikit_arg(obs)
		self.__dict__['_estimator_targets'] = targets




class ScikitEstimatorBase(Learnable, Computable, ScikitWrapper):
	
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
		targets = out.get_targets()
		if targets is None:
			super(Learnable, self).fit(out.get_observations())
		else:
			super(Learnable, self).fit(out.get_observations(), targets)
		return out


	def _call_base_sk_fn(self, key, *args, **kwargs):
		return getattr(super(ScikitWrapper, self), key, None)(*args, **kwargs)


	def predict(self, data):
		return self._predict(data)
	
	
	def _predict(self, data, key='predict'):
		return self._call_base_sk_fn(key, data)
	
	
	def predict_probs(self, data):
		return self._format_scikit_output(self._predict_probs(self._format_scikit_arg(data)))
	
	
	def _predict_probs(self, data, key='predict_proba'):
		return self._call_base_sk_fn(key, data)


	def predict_score(self, data):
		return self._format_scikit_output(self._predict_score(self._format_scikit_arg(data)))

	
	def _predict_score(self, data, key='predict_scores'):
		pred = getattr(super(ScikitWrapper, self), key, None)
		if pred is None:
			probs = self._predict_probs(data)
			if probs.shape[-1] == 2:
				return probs[:,-1]
		return pred(data)


	def predict_confidence(self, data):
		return self._predict_confidence(data)


	def _predict_confidence(self, data):
		raise NotImplementedError


	def _prediction_methods(self):
		return {'pred': self.predict, 'scores': self.predict_score, 'probs': self.predict_probs}


	def make_prediction(self, name, data):
		method = self._prediction_methods().get(name, None)
		if method is not None:
			return method(data)


	def _compute(self, run=None, dataset=None):
		if dataset is None:
			assert run is not None, 'missing dataset/run'
			dataset = run.get_dataset()
		if not self.is_fit():
			mode = dataset.get_mode()
			if mode != 'train':
				dataset.switch_to('train')
			out = self.fit(dataset)
			dataset.switch_to(mode)
		return self.evaluate(dataset, info=run)


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

		labels, pred = info.get_targets(), info.get_result('pred')

		report = metrics.classification_report(labels, pred, target_names=info.dataset.get_target_names(),
		                                       output_dict=True)
		confusion = metrics.confusion_matrix(labels, pred)

		precision, recall, fscore, support = metrics.precision_recall_fscore_support(labels, pred)

		probs = info.get_result('probs')
		# multi_class = 'ovr' if
		auc = metrics.roc_auc_score(labels, probs, multi_class='ovr')

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

		info['accuracy'] = report['accuracy']
		return info


	def get_scores(self):
		return ['accuracy', 'roc-auc', 'f1', 'precision', 'recall',
			'worst-roc-auc', 'worst-f1', 'worst-precision', 'worst-recall',
			*super().get_scores()]


	def get_results(self):
		return ['report', 'confusion',
		        'full-roc-auc', 'roc-curve',
		        'full-f1', 'full-recall', 'full-precision', 'full-support',
		        *super().get_results()]



class Regressor(Supervised, base.RegressorMixin):
	def __init__(self, A, cutoff=unspecified_argument, **kwargs):
		if cutoff is unspecified_argument:
			cutoff = A.pull('cutoff', 0.05) # in ranges for bount vars
		super().__init__(A, **kwargs)
		self._eval_cutoff = cutoff


	def _evaluate(self, info, **kwargs):
		info = super()._evaluate(info, **kwargs)

		labels, pred = info.get_targets(), info.get_result('pred')

		pred = torch.from_numpy(pred).float()
		labels = torch.from_numpy(labels).float().view(*pred.shape)

		diffs = self._outspace.difference(pred, labels)
		info.diffs = diffs
		errs = diffs.abs()

		mse = errs.pow(2).mean().item()
		mae = errs.mean().item()
		mxe = errs.max().item()
		medae = errs.median().item()

		info.update({
			'mse': mse,
			'mxe': mxe,
			'mae': mae,
			'medae': medae,
		})

		if isinstance(self._outspace, util.BoundDim):
			mx_error = self._outspace.range
		else:
			mx_error = labels.max(0)[0] - labels.min(0)[0]
		mx_error *= self._eval_cutoff
		mx_error = mx_error.unsqueeze(0)

		score = (mx_error>errs).sum() / errs.numel()
		info.score = score.item()
		return info


	def get_scores(self):
		return ['score', 'mse', 'mxe', 'mae', 'medae']



class FlatRegressor(Regressor): # topology of predicted variable is flat (not periodic)
	def _evaluate(self, info, **kwargs):
		info = super()._evaluate(info, **kwargs)

		labels, pred = info.get_targets(), info.get_result('pred')
		r2 = metrics.r2_score(labels, pred)

		info.update({
			'r2-score': r2,
			'sigma_pred': pred.std(),
			'sigma_true': labels.std(),
		})

		return info

	def get_scores(self):
		return [*super().get_scores(), 'r2-score', 'sigma_pred', 'sigma_true']



class Clustering(ScikitEstimator, base.ClusterMixin):

	def _evaluate(self, info, **kwargs):
		info = super()._evaluate(info, **kwargs)

		obs, pred = info.get_observations(), info.get_result('pred')

		silhouette = metrics.silhouette_score(obs, pred, metric='euclidean')
		calinski_harabasz = metrics.calinski_harabasz_score(obs, pred)
		davies_bouldin = metrics.davies_bouldin_score(obs, pred)

		info.update({
			'silhouette': silhouette,
			'calinski_harabasz': calinski_harabasz,
			'davies_bouldin': davies_bouldin,
		})

		true = info.get_targets()
		if true is not None:
			adjusted_rand = metrics.adjusted_rand_score(true, pred)
			ami = metrics.adjusted_mutual_info_score(true, pred)
			nmi = metrics.normalized_mutual_info_score(true, pred)
			homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(true, pred)

			true_silhouette = metrics.silhouette_score(obs, true, metric='euclidean')
			true_calinski_harabasz = metrics.calinski_harabasz_score(obs, true)
			true_davies_bouldin = metrics.davies_bouldin_score(obs, true)

			contingency = metrics.cluster.contingency_matrix(true, pred)
			pair_confusion = metrics.cluster.pair_confusion_matrix(true, pred)

			info.update({
				'adjusted_rand': adjusted_rand,
				'ami': ami,
				'nmi': nmi,
				'homogeneity': homogeneity,
				'completeness': completeness,
				'v_measure': v_measure,

				'true_silhouette': true_silhouette,
				'true_calinski_harabasz': true_calinski_harabasz,
				'true_davies_bouldin': true_davies_bouldin,

				'contingency_matrix': contingency,
				'pair_confusion': pair_confusion,
			})

			# info.score = ami

		return info


	def get_scores(self):
		return ['silhouette', 'calinski_harabasz', 'davies_bouldin',
		        'adjusted_rand', 'ami', 'nmi', 'homogeneity', 'completeness', 'v_measure',
		        'true_silhouette', 'true_calinski_harabasz', 'true_davies_bouldin',
			*super().get_scores()]


	def get_results(self):
		return ['contingency_matrix', 'pair_confusion', *super().get_results()]


	def _predict_score(self, data, key=None):
		return super()._predict_score(data, key='scores' if key is None else key)




class Outlier(ScikitEstimator, base.OutlierMixin):
	def __init__(self, A, auto_anomalous_class=unspecified_argument, auto_remove_training=None, **kwargs):
		if auto_anomalous_class is unspecified_argument:
			auto_anomalous_class = A.pull('auto-anomaly-class', None)
		if auto_remove_training is None:
			auto_remove_training = A.pull('auto-remove-training', True)
		super().__init__(A, **kwargs)
		self._auto_anomalous_class = auto_anomalous_class
		self._auto_remove_training = auto_remove_training


	def _predict_score(self, data, key=None):
		return super()._predict_score(data, key='decision_function' if key is None else key)


	def _fit(self, dataset, out=None):
		if out is None:
			out = AutomaticOutlierInfo(self, dataset, auto=self._auto_anomalous_class,
			                           remove_training=self._auto_remove_training)
		return super()._fit(dataset, out=out)


	def evaluate(self, dataset=None, info=None, out=None, **kwargs):
		if dataset is None:
			dataset = info.get_dataset()
		if out is None:
			out = AutomaticOutlierInfo(self, dataset, auto=self._auto_anomalous_class,
			                           remove_training=self._auto_remove_training)
		return super().evaluate(dataset=dataset, out=out, **kwargs)


	def _evaluate(self, info, **kwargs):
		info = super()._evaluate(info, **kwargs)

		labels, pred = info.get_targets(), info.get_result('pred')

		pred = pred<0

		scores = info.get_result('scores')
		# multi_class = 'ovr' if
		auc = metrics.roc_auc_score(labels, scores)

		scores = scores.reshape(-1)
		roc = metrics.roc_curve(labels, scores)

		info.update({
			'auroc': auc.mean(),
			'roc_curve': roc,
		})
		# info['score'] = info['auroc']
		return info


	def get_scores(self):
		return ['auroc', *super().get_scores()]


	def get_results(self):
		return ['roc_curve', *super().get_results()]

# decision_function



class Transformer(ScikitEstimator, base.TransformerMixin):
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


	def register_out_space(self, space):
		super().register_out_space(space)
		for estimator, dim in zip(self.estimators, space):
			estimator.register_out_space(dim)


	def _process_inputs(self, key, *ins):
		return len(self.estimators)*[ins]


	def _process_outputs(self, key, outs):
		if 'predict' in key:
			try:
				return torch.stack(outs, -1)
			except RuntimeError:
				return outs
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


	def _predict_score(self, data, **kwargs):
		return self._dispatch('_predict_score', data, **kwargs)


	def evaluate(self, dataset=None, info=None, **kwargs):
		if dataset is None:
			dataset = info.get_dataset()
		return self._dispatch('evaluate', dataset, **kwargs)


	def get_scores(self):
		return list({name for est in self.estimators for name in est.get_scores()})


	def get_results(self):
		return list({name for est in self.estimators for name in est.get_results()})



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
			ds.register_wrapper('single-label', idx=idx)
		return datasets


	def _process_results(self, out, filter_outputs=True):
		individuals = [estimator._process_results(res, filter_outputs=filter_outputs)
		               for estimator, res in zip(self.estimators, out)]

		merged = {}
		for scs, _ in individuals:
			for key, val in scs.items():
				if key not in merged:
					merged[key] = []
				merged[key].append(self._breakdown_val(val))

		return {key: sum(vals) / len(vals) for key, vals in merged.items()}, {'individual': individuals}




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



class Periodized(MultiEstimator, Regressor): # create a copy of the estimator for the sin component
	def __init__(self, A, cutoff=unspecified_argument, **kwargs):
		super().__init__(A, cutoff=cutoff, **kwargs)

	def _process_inputs(self, key, *ins):
		if 'fit' in key:
			infos = [ScikitEstimatorInfo(est, ins[0]) for est in self.estimators]
			infos[0]._estimator_targets, infos[1]._estimator_targets = \
				self._outspace.expand(torch.from_numpy(infos[0]._estimator_targets)).permute(2,0,1).numpy()
			return [(ins[0], info) for info in infos]
		return super()._process_inputs(key, *ins)


	def evaluate(self, dataset=None, info=None, **kwargs):
		return super(ParallelEstimator, self).evaluate(dataset=dataset, info=info, **kwargs)


	def register_out_space(self, space):
		assert isinstance(space, util.PeriodicDim)
		super().register_out_space(util.JointSpace(util.BoundDim(-1,1), util.BoundDim(-1,1)))
		self._outspace = space


	def _merge_outs(self, outs, **kwargs):
		cos, sin = outs
		theta = self._outspace.compress(torch.stack([cos, sin], -1))
		return theta



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
	
	




