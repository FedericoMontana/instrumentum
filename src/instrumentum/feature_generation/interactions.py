import collections
import logging
import numbers

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_feature_names_in,
    check_is_fitted,
)

from instrumentum.utils._decorators import timeit
from instrumentum.utils.utils import check_jobs, get_combs

logger = logging.getLogger(__name__)


class Interactions(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        operations="prod",
        degree=2,
        check_features=False,
        estimator=None,
        verbose=logging.INFO,
    ):

        self.operations = operations
        self.degree = degree
        self.check_features = check_features
        self.estimator = estimator
        self.verbose = verbose

        logger.setLevel(verbose)

    def validate_input(self):

        if isinstance(self.operations, str):
            self.operations = [self.operations]

        if not isinstance(self.operations, list):
            raise ValueError("operations error input")

        if isinstance(self.degree, numbers.Integral):
            if self.degree < 0:
                raise ValueError(
                    f"degree must be a non-negative integer, got {self.degree}."
                )
            self._max_degree = self._min_degree = self.degree
        elif (
            isinstance(self.degree, collections.abc.Iterable) and len(self.degree) == 2
        ):
            self._min_degree, self._max_degree = self.degree
            if not (
                isinstance(self._min_degree, numbers.Integral)
                and isinstance(self._max_degree, numbers.Integral)
                and self._min_degree >= 0
                and self._min_degree <= self._max_degree
            ):
                raise ValueError(
                    "degree=(min_degree, max_degree) must "
                    "be non-negative integers that fulfil "
                    "min_degree <= max_degree, got "
                    f"{self.degree}."
                )
        else:
            raise ValueError(
                "degree must be a non-negative int or tuple "
                "(min_degree, max_degree), got "
                f"{self.degree}."
            )

    @timeit
    def fit(self, X, y=None):

        self.validate_input()

        X = self._validate_data(X)
        idxs = get_combs(
            self.n_features_in_, combs_from=self._min_degree, combs_to=self._max_degree
        )

        # I prefer a tuple due to its unmutability feature (to research)
        self.interactions_ = tuple((idx, op) for idx in idxs for op in self.operations)

        logger.info(
            "Total number of interactions calculated: %s", len(self.interactions_)
        )

    def get_interaction_name(self, idxs, op):

        feat_names = _check_feature_names_in(self).take(idxs)
        return str("_" + op + "_").join(feat_names)

    def transform(self, X, y=None):

        X = self._validate_data(X, order="F", dtype=FLOAT_DTYPES, reset=False)

        # To hold the interactions
        X_ = np.empty(shape=(len(X), len(self.interactions_)), dtype=X.dtype)
        # To hold the names of the interactions
        self.names_ = np.empty(shape=len(self.interactions_), dtype=object)

        for counter, (idxs, op) in enumerate(self.interactions_):
            res = self.get_op_result(X, idxs, op)
            X_[:, counter] = res
            self.names_[counter] = self.get_interaction_name(idxs, op)

            logger.debug("New feature created: %s", self.names_[counter])

        self.names_ = np.hstack(
            (
                _check_feature_names_in(self),
                self.names_,
            )
        )

        return np.hstack((X, X_))

    def get_feature_names_out(self):
        return self.names_

    def get_op_result(self, arr, idxs, op):
        if op == "sum":
            return arr.take(idxs, axis=1).sum(axis=1)

        if op == "prod":
            return arr.take(idxs, axis=1).prod(axis=1)
