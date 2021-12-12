import logging

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from instrumentum.utils._decorators import timeit
from instrumentum.utils.utils import check_jobs

logger = logging.getLogger(__name__)


def idx_from_mask(mask):
    return np.flatnonzero(mask)  # ~ to negate


def mask_from_idx(idx, size):
    mask = np.zeros(size, dtype=bool)
    mask[idx] = True
    return mask


class ClusterSelection(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(
        self,
        method: str = "pearson",
        t: float = 0.8,
        meta_estimator=None,
        verbose=logging.INFO,
        make_matrix=None,
        criterion="distance",
    ):

        self.t = t
        self.method = method
        self.meta_estimator = meta_estimator
        self.verbose = verbose
        self.make_matrix = make_matrix
        self.criterion = criterion

        logger.setLevel(verbose)

    # TODO: agregar las opciones de distancia etc.
    def _get_clusters(self, X, y, as_dict=True):

        if self.make_matrix is not None:
            dis = self.make_matrix(X, y)
        else:
            X_corr = np.corrcoef(X, rowvar=False)  # spearmanr(X)[0]
            dis = 1 - np.fabs(X_corr)
        # shuldnt be necessary but sometimes it is not zero
        # and we get an error in squareforms
        np.fill_diagonal(dis, 0, wrap=False)
        dis = np.maximum(dis, dis.transpose())
        #
        Z = linkage(squareform(dis), "complete")

        clusters = fcluster(Z, self.t, criterion=self.criterion)

        if not as_dict:
            return clusters

        return {k: np.where(clusters == k)[0] for k in set(clusters)}

    def fit(self, X, y=None):

        X = self._validate_data(X)

        constant_mask = self._check_constants(X)
        if any(constant_mask):
            raise ValueError(
                "You must remove the constant columns before using this functionality"
            )

        self.clusters_in_ = self._get_clusters(X, y)
        self.clusters_out_ = {}

        current_mask = np.zeros(X.shape[1], dtype=bool)

        for id_cluster, id_cols in self.clusters_in_.items():

            if len(id_cols) == 1:
                logger.info(
                    "Cluster %s - Unique column: %s. Selected Automatically\n",
                    id_cluster,
                    self._get_all_features_in()[id_cols],
                )
                best_idx = id_cols
            else:
                logger.info(
                    "Cluster %s - Evaluating columns: %s",
                    id_cluster,
                    self._get_all_features_in()[id_cols],
                )

                best_idx = self._get_best_from_cluster(X, y, id_cols)

                logger.info(
                    "Columns selected: %s\n", self._get_all_features_in()[best_idx]
                )

            current_mask |= mask_from_idx(best_idx, X.shape[1])
            self.clusters_out_[id_cluster] = best_idx

        self.mask_ = current_mask

    def _get_best_from_cluster(self, X, y, id_cols):

        est = clone(self.meta_estimator)
        est.fit(X[:, id_cols], y)
        best_idx = est.get_support(indices=True)

        return id_cols[best_idx]

    def _check_constants(self, X):
        peak_to_peaks = np.ptp(X, axis=0)
        return np.isclose(peak_to_peaks, 0.0)

    def _get_all_features_in(self):
        return _check_feature_names_in(self)

    def _get_support_mask(self):
        return self.mask_

    @property
    def clusters_initial(self):
        # todo if fitted...
        return self.clusters_in_

    @property
    def clusters_final(self):
        # todo if fitted...
        return self.clusters_out_


# if __name__ == "__main__":
#     from collections import Counter

#     from sklearn.datasets import make_classification
#     from sklearn.tree import DecisionTreeClassifier

#     from instrumentum.feature_selection.correlation import ClusterSelection
#     from instrumentum.feature_selection.stepwise import DynamicStepwise
#     from instrumentum.model_tuning._optuna_dispatchers import optuna_param_disp
#     from instrumentum.model_tuning.wrapper_optuna import OptunaSearchCV
#     from sklearn.model_selection import train_test_split
#     import logging

#     import numpy as np

#     from sklearn.metrics import roc_auc_score
#     from lightgbm import LGBMClassifier

#     X, y = make_classification(
#         n_samples=2000,
#         n_redundant=5,
#         n_features=10,
#         n_informative=5,
#         random_state=111,
#     )

#     search_space = optuna_param_disp[DecisionTreeClassifier.__name__]
#     os = OptunaSearchCV(
#         estimator=DecisionTreeClassifier(),
#         scoring="roc_auc",
#         search_space=search_space,
#         n_iter=5,
#         verbose=logging.WARNING,
#         random_state=123,
#     )

#     stepw = DynamicStepwise(
#         estimator=os,
#         rounding=4,
#         n_combs=2,
#         verbose=logging.WARNING,
#         direction="forward",
#         max_cols=2,
#     )

#     threshold = 0.8

#     def score_me(X, y):
#         search_space = optuna_param_disp[LGBMClassifier.__name__]
#         os = OptunaSearchCV(
#             estimator=LGBMClassifier(),
#             scoring="roc_auc",
#             search_space=search_space,
#             n_iter=20,
#             verbose=logging.WARNING,
#             random_state=123,
#         )

#         X_train, X_test, y_train, _ = train_test_split(
#             X, y, test_size=0.25, random_state=42, stratify=y
#         )

#         return os.fit(X_train.reshape(-1, 1), y_train).predict(X_test.reshape(-1, 1))

#     def create_corr(X, y):

#         n = X.shape[1]
#         correl = np.ones((n, n), dtype=float)

#         scores = {}

#         for i in range(n - 1):
#             for j in range(i + 1, n):

#                 if i not in scores:
#                     scores[i] = score_me(X[:, i], y)

#                 if j not in scores:
#                     scores[j] = score_me(X[:, j], y)

#                 c_i = np.isclose(np.ptp(scores[i], axis=0), 0)
#                 c_j = np.isclose(np.ptp(scores[j], axis=0), 0)

#                 r = 0.5
#                 if not c_j:
#                     r = roc_auc_score(scores[j], scores[i])
#                 elif not c_i:
#                     r = roc_auc_score(scores[i], scores[j])

#                 correl[i, j] = correl[j, i] = r

#         return abs(correl - 0.5) * 2

#     # custom_corr_matrix = create_corr(X, y)
#     # custom_corr_matrix = abs(custom_corr_matrix - 0.5) * 2

#     clsl = ClusterSelection(
#         threshold=0.05, meta_estimator=stepw, make_matrix=create_corr
#     )
#     clsl.fit(X, y)

#     print("\nThese are the best columns after removal: ")
#     print(clsl.get_feature_names_out())

#     print("\nNumber of elements grouped by clusters before reduction: ")
#     print(Counter([len(v) for k, v in clsl.clusters_in_.items()]))

#     print("\nNumber of elements grouped by clusters after reduction: ")
#     print(Counter([len(v) for k, v in clsl.clusters_out_.items()]))
