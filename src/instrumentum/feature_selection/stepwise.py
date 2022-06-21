import logging

import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from instrumentum.utils._decorators import timeit
from instrumentum.utils.utils import check_jobs, get_combs

logger = logging.getLogger(__name__)


# TODO:
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
        best_candidate_strategy="round_len_score",
    ):

        self.estimator = estimator
        self.n_combs = n_combs
        self.rounding = rounding
        self.add_always = add_always
        self.direction = direction
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_cols = max_cols
        self.best_candidate_strategy = best_candidate_strategy

        logger.setLevel(verbose)

    @timeit
    def fit(self, X, y):

        if self.direction not in ("forward", "backward"):
            raise ValueError(
                "direction must be either 'forward' or 'backward'. "
                f"Got {self.direction}."
            )

        if self.best_candidate_strategy not in ("round_len_score", "score"):
            raise ValueError(
                "best_candidate_strategy must be either "
                + "'round_len_score' or 'score'. "
                f"Got {self.best_candidate_strategy}."
            )

        if not isinstance(
            self.estimator, sklearn.model_selection._search.BaseSearchCV
        ):
            raise ValueError("Estimator must implement BaseSearchCV")

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
            tracker_score = self._get_score(X[:, ~current_mask], y)
            tracker_score = round(tracker_score, self.rounding)
            logger.info("With all columns, score is %s\n", tracker_score)
        else:
            # For forward, we gotta start with 0 (no columns)
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
                    comb=comb,
                    mask=current_mask,
                )
                for comb in combs
            )

            keep_going, tracker_score, current_mask = self._update_process(
                comb_results, tracker_score, current_mask
            )

        self.mask_ = (
            current_mask if self.direction == "forward" else ~current_mask
        )

    def _get_n_combs(self, current_mask):
        if not self.max_cols:
            return self.n_combs
        # If max columns is set, and the next iterations only allow n
        # more columns, but combs > n, this will avoid testing more
        # than what max cols dictate
        n_combs_ = min(self.n_combs, self.max_cols - sum(current_mask))

        # Finishing conditions should finish the process before
        # his could happen
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
        best_comb_score, idx_best_comb_cols = self._get_best_candidate(
            comb_results
        )

        best_comb_score = round(best_comb_score, self.rounding)

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

        # When forward, we don't want to add if they dont
        # improve the score (>)
        # for backward, we prefer keep removing if they
        # don't reduce the score (>=)
        if (
            best_comb_score > tracker_score
            if fwd
            else best_comb_score >= tracker_score
        ) or self.add_always:

            current_mask[idx_best_comb_cols] = True
            tracker_score = best_comb_score
            self.seq_cols_added_.append(
                (
                    best_comb_score,
                    self._get_all_features_in()[idx_best_comb_cols],
                )
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
                "Columns were not %s as they do not improve the score."
                + "Finishing\n",
                "added" if fwd else "removed",
            )
            keep_going = False

        # Finish conditions because of limits
        if all(current_mask):
            logger.info(
                "All columns were %s. Finishing.\n",
                "added" if fwd else "removed",
            )
            keep_going = False

        elif self.max_cols and current_mask.sum() >= self.max_cols:
            logger.info("Max columns %s reached. Finishing.\n", self.max_cols)
            keep_going = False

        return keep_going, tracker_score, current_mask

    def _get_best_candidate(self, comb_results):

        if self.best_candidate_strategy == "round_len_score":

            inv = -1 if self.direction == "forward" else 1

            comb_sorted = sorted(
                comb_results,
                key=lambda x: (
                    round(x[0], self.rounding),
                    inv * len(x[1]),
                    x[0],
                ),
                reverse=True,
            )
            return comb_sorted[0]

        else:

            comb_sorted = sorted(
                comb_results,
                key=lambda x: (x[0]),
                reverse=True,
            )
            return comb_sorted[0]

    def _get_best_columns(self, X, y, comb, mask):
        logger.setLevel(self.verbose)  # needed for delayed

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
                # For backward, it might remove all columns for
                # some candidates, and won't fit.
                score = 0
            else:
                score = self._get_score(X[:, mask_candidate], y)

        logger.debug(
            "Score %s checking %s full combination: %s %s",
            score,
            self._get_all_features_in()[idx_comb_to_eval],
            self._get_all_features_in()[mask_candidate],
            "* already calculated" if saved_score else "",
        )

        return score, idx_comb_to_eval

    def _get_score(self, X, y):
        est = clone(self.estimator)
        est.fit(X, y)
        return est.best_score_

    def _get_all_features_in(self):
        return _check_feature_names_in(self)

    def _get_support_mask(self):
        return self.mask_

    @property
    def seq_columns_selected_(self):
        check_is_fitted(self)
        # How to act on this when returned (asuming ret), options:
        # (1) pd.DataFrame(ret, columns=['a', 'b']) # Pretty printer
        # (2) np.concatenate([x[1] for x in ret]) # Get all columns
        return self.seq_cols_added_
