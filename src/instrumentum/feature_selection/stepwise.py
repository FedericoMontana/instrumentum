import logging
from itertools import chain, combinations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from instrumentum.utils._decorators import timeit
from instrumentum.utils.utils import check_jobs, get_combs

logger = logging.getLogger(__name__)

# TODO:
# receive a function for scoring. And add the scoring as a method
# agregar una historia de casos para que no vuelva a verificar casos ya verificados
# Document the full shit
class DynamicStepwise(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(
        self,
        estimator,
        *,
        n_combs=1,
        rounding=4,
        add_always=False,
        direction="forward",
        n_jobs=-1,
        verbose=logging.INFO,
        # Max cols doesnt mean exactly this number, but
        # no more than this number (unless add_always is true)
        max_cols=None,
    ):

        self.estimator = estimator
        self.n_combs = n_combs
        self.rounding = rounding
        self.add_always = add_always
        self.direction = direction
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_cols = max_cols

        logger.setLevel(verbose)

    @timeit
    def fit(self, X, y):

        if self.direction not in ("forward", "backward"):
            raise ValueError(
                "direction must be either 'forward' or 'backward'. "
                f"Got {self.direction}."
            )

        # features_name_in are set if X is a DataFrame
        X, y = self._validate_data(
            X,
            y,
        )

        n_features = X.shape[1]
        self.n_jobs = check_jobs(self.n_jobs, self.verbose)
        self.seq_cols_added_ = []
        self._candidates_history = {}
        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(n_features, dtype=bool)

        if self.direction == "backward":
            est = clone(self.estimator)
            est.fit(X[:, ~current_mask], y)
            tracker_score = round(est.best_score_, self.rounding)
            logger.info("With all columns, score is %s\n", tracker_score)
        else:
            # For forward, we know we have to at least add 1 column
            tracker_score = 0

        keep_going = True

        while keep_going:

            n_cols_remaining = sum(~current_mask)
            n_combs_ = self._get_n_combs(current_mask)
            combs = get_combs(set_size=n_cols_remaining, combs_to=n_combs_)

            logger.info("Remaining columns to test: %s", n_cols_remaining)
            logger.info("Combinations to test: %s", len(combs))

            comb_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._get_best_columns)(
                    X,
                    y,
                    estimator=clone(self.estimator),
                    rounding=self.rounding,
                    comb=comb,
                    mask=current_mask,
                )
                for comb in combs
            )

            keep_going, tracker_score, current_mask = self._update_process(
                comb_results, tracker_score, current_mask
            )

        self.mask_ = current_mask if self.direction == "forward" else ~current_mask

    def _get_n_combs(self, current_mask):
        if not self.max_cols:
            return self.n_combs
        # If max columns is set, and the next iterations only allow n
        # more columns, but combs > n, this will avoid testing more
        # than what max cols dictate
        n_combs_ = min(self.n_combs, self.max_cols - sum(current_mask))

        # Finishing conditions should finish the process before this could happen
        assert n_combs_ > 0

        return n_combs_

    def _update_process(self, comb_results, tracker_score, current_mask):

        assert len(comb_results) > 0
        # easier to query
        fwd = self.direction == "forward"

        # Add to the history of candidates
        self._candidates_history.update(
            {
                frozenset(np.append(comb, np.flatnonzero(current_mask))): score
                for score, comb in comb_results
            }
        )

        # Get the best ccandidate
        best_comb_score, idx_best_comb_cols = max(
            comb_results, key=lambda item: item[0]
        )

        logger.info(
            "Best score from combinations: %s, global score %s",
            best_comb_score,
            tracker_score,
        )

        logger.info(
            "Best score comes from %s columns: %s",
            "adding" if fwd else "removing",
            self._get_all_features_in()[idx_best_comb_cols],
        )

        # When forward, we don't want to add if they dont improve the score (>)
        # for backward, we prefer keep removing if they don't reduce the score (>=)
        if (
            best_comb_score > tracker_score if fwd else best_comb_score >= tracker_score
        ) or self.add_always:

            current_mask[idx_best_comb_cols] = True
            tracker_score = best_comb_score
            self.seq_cols_added_.append(
                (best_comb_score, self._get_all_features_in()[idx_best_comb_cols])
            )

            logger.info(
                "Best columns were %s. All columns %s so far %s\n",
                "added" if fwd else "removed",
                "added" if fwd else "removed",
                self._get_all_features_in()[current_mask],
            )
            keep_going = True

        else:
            logger.info(
                "Columns were not %s as they do not improve the score. Finishing\n",
                "added" if fwd else "removed",
            )
            keep_going = False

        # Finish conditions because of limits
        if all(current_mask):
            logger.info(
                "All columns were %s. Finishing.\n", "added" if fwd else "removed"
            )
            keep_going = False

        elif self.max_cols and current_mask.sum() >= self.max_cols:
            logger.info("Max columns %s reached. Finishing.\n", self.max_cols)
            keep_going = False

        return keep_going, tracker_score, current_mask

    def _get_best_columns(self, X, y, estimator, rounding, comb, mask):
        logger.setLevel(self.verbose)  # needed for delayed

        try:
            check_is_fitted(estimator)
        except NotFittedError:
            pass
        else:
            raise ValueError("Estimator was fitted already")

        idx_not_processed = np.flatnonzero(~mask)
        idx_comb_to_eval = np.take(idx_not_processed, comb)

        mask_candidate = mask.copy()
        mask_candidate[idx_comb_to_eval] = True

        if self.direction == "backward":
            mask_candidate = ~mask_candidate

        # Did we calculated this already?
        saved_score = self._candidates_history.get(
            frozenset(np.flatnonzero(mask_candidate)), None
        )

        if saved_score:
            score = saved_score
        else:
            if sum(mask_candidate) == 0:
                # For backward, it might remove all columns for some candidates, and won't fit.
                score = 0
            else:
                estimator.fit(X[:, mask_candidate], y)
                score = round(estimator.best_score_, rounding)

        logger.debug(
            "Score %s checking %s full combination: %s %s",
            score,
            self._get_all_features_in()[idx_comb_to_eval],
            self._get_all_features_in()[mask_candidate],
            "* already calculated" if saved_score else "",
        )

        return score, idx_comb_to_eval

    # this is an utility function to get the names of the
    # columns passed
    # it works with numpy and dataframes (leverage some
    # sklearn functionality).
    def _get_all_features_in(self):
        return _check_feature_names_in(self)

    def _get_support_mask(self):
        return self.mask_

    @property
    def seq_columns_selected_(self):
        # todo if fitted...
        # How to act on this when returned (asuming ret), options:
        # (1) pd.DataFrame(ret, columns=['a', 'b']) # Pretty printer
        # (2) np.concatenate([x[1] for x in ret]) # Get all columns
        return self.seq_cols_added_
