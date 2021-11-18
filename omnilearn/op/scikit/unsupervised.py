

from omnibelt import unspecified_argument
import omnifig as fig

from sklearn.ensemble import IsolationForest as SK_IsolationForest
from sklearn.cluster import KMeans as SK_KMeans

from .wrappers import SingleLabelEstimator, Regressor, Classifier, Clustering, Outlier


@fig.Component('kmeans')
class KMeans(Clustering, SK_KMeans):
	def __init__(self, A, n_clusters=None, init=None, n_init=None, max_iter=None, tol=None, verbose=None,
	             random_state=unspecified_argument, copy_x=None, algorithm=None, **kwargs):

		if n_clusters is None:
			n_clusters = A.pull('n-clusters', 8)

		if init is None:
			init = A.pull('init', 'k-means++')

		if n_init is None:
			n_init = A.pull('n_init', 10)

		if max_iter is None:
			max_iter = A.pull('max_iter', 300)

		if tol is None:
			tol = A.pull('tol', 0.0001)

		if verbose is None:
			verbose = A.pull('verbose', False)

		if random_state is unspecified_argument:
			random_state = A.pull('seed', None)

		if copy_x is None:
			copy_x = A.pull('copy_x', True)

		if algorithm is None:
			algorithm = A.pull('algorithm', 'auto')

		super().__init__(A, n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
		                 verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm, **kwargs)



@fig.Component('isolation-forest')
class IsolationForest(Outlier, SK_IsolationForest):
	def __init__(self, A, n_estimators=None, max_samples=None, contamination=None, max_features=None,
	             bootstrap=None, n_jobs=unspecified_argument, random_state=unspecified_argument,
	             verbose=unspecified_argument, warm_start=None, **kwargs):

		if n_estimators is None:
			n_estimators = A.pull('n-estimators', 100)

		if max_samples is None:
			max_samples = A.pull('max_samples', 'auto')

		if contamination is None:
			contamination = A.pull('contamination', 'auto')

		if max_features is None:
			max_features = A.pull('max_features', 1.)

		if bootstrap is None:
			bootstrap = A.pull('bootstrap', False)

		if verbose is unspecified_argument:
			verbose = A.pull('verbose', 0)

		if random_state is unspecified_argument:
			random_state = A.pull('seed', None)

		if n_jobs is unspecified_argument:
			n_jobs = A.pull('n_jobs', None)

		if warm_start is None:
			warm_start = A.pull('warm_start', False)

		super().__init__(A, n_estimators=n_estimators, max_samples=max_samples, contamination=contamination,
		                 max_features=max_features, bootstrap=bootstrap, verbose=verbose, random_state=random_state,
		                 n_jobs=n_jobs, warm_start=warm_start, **kwargs)








