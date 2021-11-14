
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_is_fitted, _check_feature_names_in
from sklearn.exceptions import NotFittedError

from instrumentum.utils.validation import check_jobs

from itertools import combinations
from itertools import chain
from joblib import Parallel, delayed

import logging
logger = logging.getLogger(__name__)


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
    ):

        self.estimator = estimator
        self.n_combs = n_combs
        self.rounding = rounding
        self.add_always = add_always
        self.direction = direction
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        logger.setLevel(verbose)

    def fit(self, X, y):
    
        # features_name_in are set if X is a DataFrame
        X = self._validate_data(
            X,
        )
        
        n_features = X.shape[1]
        self.n_jobs = check_jobs(self.n_jobs, self.verbose)
        self.seq_cols_added_ = []
        
        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(n_features, dtype=bool)
 
        tracker_score = 0
        keep_going = True

        while keep_going:

            n_cols_remaining = sum(~current_mask)
            combs = list(self._get_combs(n_cols_remaining, self.n_combs))

            logger.info("Remaining columns to test: %s", n_cols_remaining)
            logger.info("Combinations to test: %s", len(combs))

            comb_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._get_best_columns)(
                    X,
                    y,
                    estimator=clone(self.estimator),
                    rounding=self.rounding,
                    comb=list(comb),
                    mask=current_mask,
                )
                for comb in combs
            )

            best_comb_score, idx_best_comb_cols = max(comb_results, key=lambda item: item[0])
            current_mask[idx_best_comb_cols] = True
            
            logger.info(
                "Best score from combinations: %s, global score %s",
                best_comb_score,
                tracker_score,
            )
            logger.info("Best score comes from columns: %s", self._get_all_features_in()[idx_best_comb_cols])

            # Check if finish conditions are met
            if best_comb_score > tracker_score or self.add_always:
                tracker_score = best_comb_score
                
                logger.info(
                    "Best columns were added. All columns added so far %s\n", self._get_all_features_in()[current_mask]
                )
                self.seq_cols_added_.append((best_comb_score, self._get_all_features_in()[idx_best_comb_cols]))

            else:
                logger.info(
                    "Columns were not added as they do not increase score. Finishing\n"
                )
                keep_going = False

            if all(current_mask):
                logger.info("All columns were added. Finishing\n")
                keep_going = False
            
        self.mask_ = current_mask

    def _get_combs(self, set_size, combs, include_empty=False):

        l_comb = [
            combinations(list(range(0, set_size)), x)
            for x in range((0 if include_empty else 1), combs + 1)
        ]

        return chain.from_iterable(l_comb)

    def _get_best_columns(self, X, y, estimator, rounding, comb, mask):
        logger.setLevel(self.verbose)
        
        try:
            check_is_fitted(estimator)
        except NotFittedError:
            pass
        else:
            raise ValueError("Estimator was fitted already")
        
        idx_not_processed = np.flatnonzero(~mask)
        
        # Comb has the index to select FROM the index of 
        # columns not yet evaluated
        idx_comb_to_eval = idx_not_processed[comb]
        idx_candidate = np.append(np.flatnonzero(mask), idx_comb_to_eval)
        estimator.fit(X[:,idx_candidate], y)
        score = round(estimator.best_score_, rounding)
        logger.debug("Score %s for this combination: %s", score, self._get_all_features_in()[idx_candidate])

        return score, idx_comb_to_eval


    # this is an utility function to get the names
    # it works with numpy and dataframes (leverage some
    # sklearn functionality).
    def _get_all_features_in(self):
        return _check_feature_names_in(self)
        
    def _get_support_mask(self):
        return self.mask_
    
    @property
    def seq_cols_processed(self):
        #todo if fitted...
        # How to act on this when returned (asuming ret)
        # (1) pd.DataFrame(ret, columns=['a', 'b']) # Pretty printer
        # (2) np.concatenate([x[1] for x in ret]) # Get all columns
        return self.seq_cols_added_
# ------------------------------------------------------ #

import pandas as pd
from instrumentum.model_tuning._optuna_dispatchers import optuna_param_disp
from instrumentum.model_tuning.wrapper_optuna import OptunaSearchCV as opcv

from sklearn.datasets import make_classification

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import VarianceThreshold

from instrumentum.feature_selection.swcopy import DynamicStepwise as dsw
import numpy as np

if __name__ == "__main__":
    
    # X, y = make_classification(n_samples=5000, n_features=6, n_informative=3, random_state=111)
    # X = pd.DataFrame(X)
    # pd.Series(y)
    
    # est = DecisionTreeClassifier()
    # # est.fit(X, y)
    # # print(est.feature_names_in_)
    
    # # X = np.array([[1, 23, 3], [1, 534, 3]], np.int32)
    # # print(X.dtype.names)
    # X = pd.DataFrame(X)
    # y = pd.Series(y)
    
    data_file = "/Users/federico/codes/instrumentum/docs/sample_data/simple.csv"
    data_df = pd.read_csv(data_file)
    data_df['target'] = data_df['target'].replace([-1],0)
    #data_df = data_df[['a', 'b', 'c', 'd', 'e', 'target']]
    
    X = data_df.drop("target",axis=1).values
    y = data_df['target']

     
    search_space = optuna_param_disp[DecisionTreeClassifier.__name__]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=123)
    os = opcv(estimator=DecisionTreeClassifier(), scoring = "roc_auc", cv=cv, search_space=search_space, n_iter=2, random_state=123)
  
    stepw = dsw(estimator=os, n_combs=2, verbose=logging.INFO)
    
    stepw.fit(X,y)
    print(stepw.seq_cols_added_)
    # vs = VarianceThreshold(0.99)
    #vs.fit(X)
    # print(vs.get_feature_names_out())
#     print(X.shape)
#     print(vs.transform(X))
#     # search_space = optuna_param_disp[DecisionTreeClassifier.__name__]
#     # cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2)
    
#     # os = opcv(estimator=DecisionTreeClassifier(), scoring = "roc_auc", cv=cv, search_space=search_space, n_iter=2)

#     # os.fit(X, y)
#     # print("Scorer: ", os.scorer_)
#     # print(os.best_params_)
#     # print(os.best_score_)
#     # print("Scoring base: ", os.score(X, y))