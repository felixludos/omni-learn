import copy

from omnibelt import agnosticmethod
import torch
import torch.multiprocessing as mp
import numpy as np
from sklearn import metrics
# from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from omniplex.structure import spaces, Function
from .models import Model


class AbstractScikitEstimator(Model, Function):
	class ResultsContainer(Model.ResultsContainer):
		def __init__(self, estimator=None, **kwargs):
			super().__init__(**kwargs)
			self.estimator = estimator


		class UnknownResultError(KeyError):
			pass


		def get_result(self, key, **kwargs):
			# return self._find_missing(key)
			if key not in self:
				if self.estimator is None or key not in self.estimator.prediction_methods():
					raise self.UnknownResultError(key)
				self[key] = self._infer(key, **kwargs)
			return self[key]


		def _infer(self, key, observation=None, **kwargs):
			if observation is None:
				observation = self['observation']
			return self.estimator.prediction_methods()[key](observation)


		def _find_missing(self, key, **kwargs):
			if self.estimator is not None and key in self.estimator.prediction_methods():
				return self.get_result(key, **kwargs)
			return super()._find_missing(key, **kwargs)


	@agnosticmethod
	def create_results_container(self, info=None, **kwargs):
		return super().create_results_container(estimator=None if type(self) == type else self, info=info, **kwargs)



	def prediction_methods(self):
		return {}


	@agnosticmethod
	def heavy_results(self):
		return {'observation', *super().heavy_results()}


	@staticmethod
	def _format_scikit_arg(data):
		if data is not None and isinstance(data, torch.Tensor):
			data = data.detach().cpu().numpy()
		return data


	@staticmethod
	def _format_scikit_output(out):
		if out is not None and isinstance(out, np.ndarray):
			out = torch.from_numpy(out)
		return out


	@agnosticmethod
	def _get_scikit_fn(self, key):
		raise NotImplementedError


	@agnosticmethod
	def _call_scikit_fn(self, fn, *args):
		if isinstance(fn, str):
			fn = self._get_scikit_fn(fn)
		return self._format_scikit_output(fn(*[self._format_scikit_arg(arg) for arg in args]))



class AbstractSupervised(AbstractScikitEstimator): # just a flag to unify wrappers and nonwrappers
	def prediction_methods(self):
		return {'pred': self.predict}

	@agnosticmethod
	def heavy_results(self):
		return {'target', 'pred', *super().heavy_results()}

	def predict(self, observation, **kwargs):
		return self._call_scikit_fn('predict', observation).view(observation.shape[0], -1)


	def _fit(self, info, observation=None, target=None):
		if observation is None:
			observation = info['observation']
		if target is None:
			target = info['target']
		self._call_scikit_fn('fit', observation, target.squeeze())
		return self._evaluate(info)
		return info



class ScikitEstimator(AbstractScikitEstimator):
	@agnosticmethod
	def _get_scikit_fn(self, key):
		return getattr(super(Model, self), key)



class ScikitEstimatorWrapper(AbstractScikitEstimator):
	def __init__(self, estimator, **kwargs):
		super().__init__(**kwargs)
		self.base_estimator = estimator


	@agnosticmethod
	def _get_scikit_fn(self, key):
		return getattr(self.base_estimator, key)



class Regressor(AbstractSupervised):
	score_key = 'r2'

	def __init__(self, *args, standardize_target=True, success_threshold=0.1, **kwargs):
		super().__init__(*args, **kwargs)
		self.standardize_target = standardize_target
		self.success_threshold = success_threshold


	def predict(self, observation, **kwargs):
		pred = super().predict(observation, **kwargs)
		if self.standardize_target:
			pred = self.dout.unstandardize(pred)
		return pred


	def _fit(self, info, observation=None, target=None):
		if target is None:
			target = info['target']
		if self.standardize_target:
			dout = info.source.space_of('target')
			target = dout.standardize(target)
		return super()._fit(info, observation=observation, target=target)


	@agnosticmethod
	def heavy_results(self):
		return {'errs', *super().heavy_results()}

	@agnosticmethod
	def score_names(self):
		return {'mse', 'mxe', 'mae', 'medae', *super().score_names()}

	def _evaluate(self, info):
		dout = info.source.space_of('target')

		target, pred = info['target'], info['pred']
		# if self.standardize_target:
		# 	target, pred = dout.standardize(target), dout.standardize(pred)

		# pred = torch.from_numpy(pred).float()
		# labels = torch.from_numpy(labels).float().view(*pred.shape)

		diffs = dout.difference(pred, target)
		info.diffs = diffs
		errs = diffs.abs()

		mse = errs.pow(2).mean().item()
		mae = errs.mean().item()
		mxe = errs.max().item()
		medae = errs.median().item()

		info.update({
			'error': errs,

			'mse': mse,
			'mxe': mxe,
			'mae': mae,
			'medae': medae,
		})

		# relative to prior
		if isinstance(dout, spaces.Bound):
			mx_error = dout.range
			if isinstance(dout, spaces.Periodic):
				mx_error /= 2
			avg_error = mx_error / 2 # assuming a uniform distribution
		else:
			mx_error = target.max(0)[0] - target.min(0)[0]
			avg_error = target.std(0) # assuming a normal distribution
		mx_error, avg_error = mx_error.view(1, -1), avg_error.view(1, -1)

		info['r2'] = 1 - errs.mean(0, keepdim=True).div(avg_error).mean().item()
		if self.success_threshold is not None:
			cutoff = avg_error * self.success_threshold
			info['success'] = cutoff.sub(errs).ge(0).prod(-1).sum().item() / errs.shape[0]
		return info



class Classifier(AbstractSupervised):
	score_key = 'f1'


	def predict_probs(self, observation, **kwargs):
		return self._call_scikit_fn('predict_proba', observation)


	def prediction_methods(self):
		methods = super().prediction_methods()
		methods['probs'] = self.predict_probs
		return methods

	@agnosticmethod
	def heavy_results(self):
		return {'probs', *super().heavy_results()}

	@agnosticmethod
	def score_names(self):
		return {'accuracy', 'roc-auc', 'f1', 'precision', 'recall',
		        'worst-roc-auc', 'worst-f1', 'worst-precision', 'worst-recall',
		        *super().score_names()}


	def _evaluate(self, info, **kwargs):
		dout = info.source.space_of('target')

		target, pred = info['target'], info['pred']
		target_, pred_ = self._format_scikit_arg(target.squeeze()), self._format_scikit_arg(pred.squeeze())

		names = getattr(dout, 'values', list(map(str, range(dout.n))))
		report = metrics.classification_report(target_, pred_, target_names=names, output_dict=True)
		confusion = metrics.confusion_matrix(target_, pred_)

		precision, recall, fscore, support = metrics.precision_recall_fscore_support(target_, pred_)

		probs_ = self._format_scikit_arg(info['probs'].squeeze())
		if probs_.shape[1] == 2:
			probs_ = probs_[:,0]
		# multi_class = 'ovr' if
		auc = metrics.roc_auc_score(target_, probs_, multi_class='ovr')

		roc = None
		if dout.n == 2:
			roc = metrics.roc_curve(target_, probs_)

		info.update({
			'roc-auc': auc.mean().item(),
			'f1': fscore.mean().item(),
			'precision': precision.mean().item(),
			'recall': recall.mean().item(),

			'worst-roc-auc': auc.min().item(),
			'worst-f1': fscore.min().item(),
			'worst-precision': precision.min().item(),
			'worst-recall': recall.min().item(),

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



class ParallelEstimator(AbstractScikitEstimator):
	def __init__(self, estimators, pbar=None, num_workers=None, **kwargs):
		super().__init__(**kwargs)
		self.estimators = estimators
		self.pbar = pbar
		self.num_workers = num_workers


	def __getitem__(self, item):
		return self.estimators[item]


	def include_estimators(self, *estimators):
		self.estimators.extend(estimators)


	def _process_inputs(self, key, *ins, **kwargs):
		return len(self.estimators)*[ins]


	def _process_outputs(self, key, outs):
		try:
			return torch.stack(outs, 1)
		except TypeError:
			return outs


	@staticmethod
	def _dispatch_worker(estimator, key, inp):
		return getattr(estimator, key)(*inp)


	def _dispatch(self, key, *ins, **kwargs):
		ins = self._process_inputs(key, *ins)
		itr = zip(self.estimators, [key]*len(self.estimators), ins)
		if self.pbar is not None:
			itr = self.pbar(itr, total=len(self.estimators), desc=key)
		if self.num_workers is None:
			outs = [self._dispatch_worker(estimator, key, inp) for estimator, key, inp in itr]
		else:
			with mp.Pool(self.num_workers) as pool:
				outs = pool.map(self._dispatch_worker, itr)
		return self._process_outputs(key, outs)


	def _merge_results(self, infos):
		info = self.create_results_container()
		info['individuals'] = infos
		try:
			scores = [i['score'] for i in infos]
			info['score'] = np.mean(scores).item()
			# info['indivdiuals'] = infos
		except KeyError:
			pass
		return info


	def filter_heavy(self, info):
		out = super().filter_heavy(info)
		if 'individuals' in info:
			out['individuals'] = [est.filter_heavy(inf) for est, inf in zip(self, info['individuals'])]
		return out


	def fit(self, source, **kwargs):
		infos = self._dispatch('fit', source, **kwargs)
		return self._merge_results(infos)


	def evaluate(self, source, **kwargs):
		infos = self._dispatch('evaluate', source, **kwargs)
		return self._merge_results(infos)


	def predict(self, data, **kwargs):
		return self._dispatch('predict', data, **kwargs)


	def predict_probs(self, data, **kwargs):
		return self._dispatch('predict_probs', data, **kwargs)



class JointEstimator(ParallelEstimator): # collection of single dim estimators (can be different spaces)
	def __init__(self, estimators, **kwargs):
		super().__init__(estimators, **kwargs)
		for estimator, dout in zip(self.estimators, self.dout):
			estimator.dout = dout


	def _process_inputs(self, key, *ins, **kwargs):
		if key in {'fit', 'evaluate'}:
			return self._split_source(key, *ins, **kwargs)
		return super()._process_inputs(key, *ins, **kwargs)


	_split_key = 'target'
	def _split_source(self, key, source, split_key=None):
		if split_key is None:
			split_key = self._split_key

		target = source[split_key]

		dims = source.space_of(split_key)

		sources = []
		start = 0
		for dim in dims:
			view = source.create_view()
			view.register_buffer(split_key, target.narrow(1, start, len(dim)), space=dim)
			sources.append((view,))
			start += len(dim)
		return sources



class MultiEstimator(ParallelEstimator): # all estimators must be of the same kind (out space)
	def __init__(self, raw_pred=False, **kwargs):
		super().__init__(**kwargs)
		self._raw_pred = raw_pred


	def toggle_raw_pred(self, val=None):
		if val is None:
			val = not self._raw_pred
		self._raw_pred = val


	def __getattr__(self, item):
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return getattr(self.estimators[0], item)


	def _merge_predictions(self, outs, **kwargs):
		raise NotImplementedError


	def _process_outputs(self, key, outs, **kwargs):
		if 'predict' in key:
			return outs if self._raw_pred else self._merge_predictions(outs, **kwargs)
		return super()._process_outputs(key, outs)



class Periodized(MultiEstimator, Regressor): # create a copy of the estimator for the sin component
	def __init__(self, estimators=None, **kwargs):
		super().__init__(estimators=estimators, **kwargs)
		assert len(estimators) == 2, 'must have 2 estimators for cos and sin' # TODO: use dedicated exception type


	def _prep_estimator(self, estimator): # TODO: maybe remove or fix deep copy
		assert estimator is not None, 'No base estimator provided'
		estimators = [estimator, copy.deepcopy(estimator)]
		return estimators


	_split_key = 'target'
	def _split_source(self, key, source, split_key=None):
		if split_key is None:
			split_key = self._split_key
		target = source[split_key]

		cos, sin = self.dout.expand(target).permute(2,0,1)

		cos_source = source.create_view()
		sin_source = source.create_view()

		cos_source.register_buffer(split_key, cos, space=self.estimators[0].dout)
		sin_source.register_buffer(split_key, sin, space=self.estimators[1].dout)
		return [(cos_source,), (sin_source,)]


	def _process_inputs(self, key, *ins, **kwargs):
		if key in {'fit', 'evaluate'}:
			return self._split_source(key, *ins, **kwargs)
		return super()._process_inputs(key, *ins, **kwargs)


	def evaluate(self, source, online=False, **kwargs):
		raws = None if online else super().evaluate(source)
		out = super(ParallelEstimator, self).evaluate(source, **kwargs)
		if raws is not None:
			out['individuals'] = raws['individuals']
		return out


	def _merge_predictions(self, outs, **kwargs):
		cos, sin = outs
		theta = self.dout.compress(torch.stack([cos, sin], -1))
		return theta



