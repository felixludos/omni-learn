
from omnibelt import unspecified_argument
import omnifig as fig

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from .wrappers import SingleLabelEstimator

# A.push('default-regressor', 'gbt-regressor', overwrite=False)
# A.push('default-classifier', 'gbt-classifier', overwrite=False)

@fig.Component('gbt-regressor')
class GBTRegressor(SingleLabelEstimator, GradientBoostingRegressor):
	def __init__(self, A, loss=None, learning_rate=None, n_estimators=None,
	             subsample=None, criterion=None, min_samples_split=None,
	             min_samples_leaf=None, min_weight_fraction_leaf=None, max_depth=None, min_impurity_decrease=None,
	             min_impurity_split=unspecified_argument, init=unspecified_argument, random_state=unspecified_argument,
	             max_features=unspecified_argument, alpha=None, verbose=None, max_leaf_nodes=unspecified_argument,
	             warm_start=None, validation_fraction=None, n_iter_no_change=unspecified_argument, tol=None,
	             ccp_alpha=None, **kwargs):

		if loss is None:
			loss = A.pull('gb_loss', '<>loss', 'ls')

		if learning_rate is None:
			learning_rate = A.pull('gb_learning_rate', '<>learning_rate', 0.1)

		if n_estimators is None:
			n_estimators = A.pull('n_estimators', 100)

		if subsample is None:
			subsample = A.pull('subsample', 1.)

		if criterion is None:
			criterion = A.pull('gb_criterion', 'friedman_mse')

		if min_samples_split is None:
			min_samples_split = A.pull('min_samples_split', 2)

		if min_samples_leaf is None:
			min_samples_leaf = A.pull('min_samples_leaf', 1)

		if min_weight_fraction_leaf is None:
			min_weight_fraction_leaf = A.pull('min_weight_fraction_leaf', 0.)

		if max_depth is None:
			max_depth = A.pull('max_depth', 3)

		if min_impurity_decrease is None:
			min_impurity_decrease = A.pull('max_impurity_decrease', 0.)

		if min_impurity_split is unspecified_argument:
			min_impurity_split = A.pull('min_impurity_split', None)

		if init is unspecified_argument:
			init = A.pull('init', None)

		if random_state is unspecified_argument:
			random_state = A.pull('random_state', None)

		if max_features is unspecified_argument:
			max_features = A.pull('max_features', None)

		if alpha is None:
			alpha = A.pull('alpha', 0.9)

		if verbose is None:
			verbose = A.pull('verbose', False)

		if max_leaf_nodes is unspecified_argument:
			max_leaf_nodes = A.pull('max_leaf_nodes', None)

		if warm_start is None:
			warm_start = A.pull('warm_start', False)

		if validation_fraction is None:
			validation_fraction = A.pull('validation_fraction', 0.1)

		if n_iter_no_change is unspecified_argument:
			n_iter_no_change = A.pull('n_iter_no_change', None)

		if tol is None:
			tol = A.pull('tolance', '<>tol', 0.0001)

		if ccp_alpha is None:
			ccp_alpha = A.pull('ccp_alpha', 0.)

		super().__init__(A, loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
		                 criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
		                 min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth,
		                 min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, init=init,
	                     random_state=random_state, max_features=max_features, alpha=alpha, verbose=verbose,
	                     max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction,
	                     n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha, **kwargs)



@fig.Component('gbt-classifier')
class GBTClassifier(SingleLabelEstimator, GradientBoostingClassifier):
	def __init__(self, A , loss=None, learning_rate=None, n_estimators=None, subsample=None, criterion=None,
	             min_samples_split=None, min_samples_leaf=None, min_weight_fraction_leaf=None, max_depth=None,
	             min_impurity_decrease=None, min_impurity_split=unspecified_argument, init=unspecified_argument,
	             random_state=unspecified_argument, max_features=unspecified_argument, verbose=None,
	             max_leaf_nodes=unspecified_argument, warm_start=None, validation_fraction=None,
	             n_iter_no_change=unspecified_argument, tol=None, ccp_alpha=None, **kwargs):

		if loss is None:
			loss = A.pull('gb_loss', '<>loss', 'deviance')

		if learning_rate is None:
			learning_rate = A.pull('gb_learning_rate', '<>learning_rate', 0.1)

		if n_estimators is None:
			n_estimators = A.pull('n_estimators', 100)

		if subsample is None:
			subsample = A.pull('subsample', 1.)

		if criterion is None:
			criterion = A.pull('gb_criterion', 'friedman_mse')

		if min_samples_split is None:
			min_samples_split = A.pull('min_samples_split', 2)

		if min_samples_leaf is None:
			min_samples_leaf = A.pull('min_samples_leaf', 1)

		if min_weight_fraction_leaf is None:
			min_weight_fraction_leaf = A.pull('min_weight_fraction_leaf', 0.)

		if max_depth is None:
			max_depth = A.pull('max_depth', 3)

		if min_impurity_decrease is None:
			min_impurity_decrease = A.pull('max_impurity_decrease', 0.)

		if min_impurity_split is unspecified_argument:
			min_impurity_split = A.pull('min_impurity_split', None)

		if init is unspecified_argument:
			init = A.pull('init', None)

		if random_state is unspecified_argument:
			random_state = A.pull('random_state', None)

		if max_features is unspecified_argument:
			max_features = A.pull('max_features', None)

		if verbose is None:
			verbose = A.pull('verbose', False)

		if max_leaf_nodes is unspecified_argument:
			max_leaf_nodes = A.pull('max_leaf_nodes', None)

		if warm_start is None:
			warm_start = A.pull('warm_start', False)

		if validation_fraction is None:
			validation_fraction = A.pull('validation_fraction', 0.1)

		if n_iter_no_change is unspecified_argument:
			n_iter_no_change = A.pull('n_iter_no_change', None)

		if tol is None:
			tol = A.pull('tolance', '<>tol', 0.0001)

		if ccp_alpha is None:
			ccp_alpha = A.pull('ccp_alpha', 0.)

		super().__init__(A, loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
		                 criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
		                 min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth,
		                 min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, init=init,
		                 random_state=random_state, max_features=max_features, verbose=verbose,
		                 max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction,
		                 n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha, **kwargs)







