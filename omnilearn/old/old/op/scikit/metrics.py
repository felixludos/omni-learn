#
# from sklearn.metrics import accuracy_score, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, \
# 	cohen_kappa_score, confusion_matrix, dcg_score, det_curve, f1_score, fbeta_score, hamming_loss, hinge_loss, \
# 	jaccard_score, log_loss, matthews_corrcoef, multilabel_confusion_matrix, ndcg_score, precision_score, \
# 	precision_recall_fscore_support, recall_score, roc_auc_score, roc_curve, top_k_accuracy_score, zero_one_loss, \
# 	precision_recall_curve, \
# 	 \
# 	explained_variance_score, max_error, mean_squared_error, mean_absolute_error, mean_squared_log_error, \
# 	median_absolute_error, mean_absolute_percentage_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, \
# 	mean_tweedie_deviance, \
# 	 \
# 	adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, \
# 	completeness_score, fowlkes_mallows_score, homogeneity_completeness_v_measure, mutual_info_score, \
# 	normalized_mutual_info_score, rand_score, silhouette_score, silhouette_samples, v_measure_score
# from sklearn.metrics.cluster import consensus_score, contingency_matrix
#
# from .wrappers import Computable
#
#
# _classification_metrics = {
# 	'accuracy': accuracy_score,
# 	'average-precision': average_precision_score,
# 	'balanced-accuracy': balanced_accuracy_score,
# 	'brier-score': brier_score_loss,
# 	'cohen-kappa': cohen_kappa_score,
# 	'confusion-matrix': confusion_matrix,
# 	'dcg-score': dcg_score,
# 	'det-curve': det_curve,
# 	'f1': f1_score,
# 	'fbeta': fbeta_score,
# 	'hamming': hamming_loss,
# 	'hinge': hinge_loss,
# 	'jaccard': jaccard_score,
# 	'log-loss': log_loss,
# 	'matthews-cor': matthews_corrcoef,
# 	'multilabel-confusion-matrix': multilabel_confusion_matrix,
# 	'ndcg': ndcg_score,
# 	'precision-recall-curve': precision_recall_curve,
# 	'precision-recall-fscore': precision_recall_fscore_support,
# 	'precision': precision_score,
# 	'recall': recall_score,
# 	'roc-auc': roc_auc_score,
# 	'roc': roc_curve,
# 	'top-k-accuracy': top_k_accuracy_score,
# 	'zero-one': zero_one_loss,
# }
#
# class AccuracyScore(Computable):
# 	_name = 'accuracy'
# 	_fn = accuracy_score
# 	_arg_reqs = {'y_true':'labels'}
#
#
#
#
