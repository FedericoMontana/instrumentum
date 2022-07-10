import logging

import numpy as np
import optuna
import sklearn
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class OptunaSearchCV(BaseSearchCV):

    _required_parameters = ["estimator", "search_space"]

    def __init__(
        self,
        estimator,
        search_space,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=logging.INFO,
        random_state=None,
        pipe_estimator=None,
    ):
        self.search_space = search_space
        self.n_iter = n_iter
        self.random_state = random_state
        self.pipe_estimator = pipe_estimator

        logger.setLevel(verbose)

        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=0,
            pre_dispatch=None,
            error_score=np.nan,
            return_train_score=False,
        )

    def fit(self, X, y):

        # Validations
        if callable(self.scoring):
            scorer = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorer = check_scoring(self.estimator, self.scoring)
        else:
            raise ValueError("Error in parameter scoring")

        if isinstance(self.estimator, sklearn.pipeline.Pipeline) ^ bool(
            self.pipe_estimator
        ):
            raise ValueError(
                "Estimator must be a Pipeline and pipe_estimator supplied"
            )

        cv_orig = check_cv(
            self.cv, y, classifier=is_classifier(self.estimator)
        )

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._opt_generic_objective(
                trial=trial,
                X=X,
                y=y,
                cv=cv_orig,
                scoring=scorer,
            ),
            n_trials=self.n_iter,
        )

        # Let's set some values so the upper classes can work
        if self.refit:
            self.best_estimator_ = self._get_estimator(**study.best_params)
            self.best_estimator_.fit(X, y)

        self.scorer_ = scorer  # important so it knows how to calculate score
        self.study_ = study
        self.multimetric_ = None  # for compatibility

        return self

    def _get_estimator(self, **param):

        if self.random_state is not None:
            param["random_state"] = self.random_state

        # if the estimator is within a pipeline, in order to access it
        # with set_params, we need to append the name of that estimator
        # within the pipeline
        if self.pipe_estimator is not None:
            param = {
                self.pipe_estimator + "__" + k: v for k, v in param.items()
            }

        clf = clone(clone(self.estimator).set_params(**param))

        return clf

    def _opt_generic_objective(self, trial, X, y, cv, scoring):

        param = self.search_space(trial)

        candidate_estimator = self._get_estimator(**param)

        score = cross_val_score(
            candidate_estimator, X=X, y=y, cv=cv, scoring=scoring
        ).mean()

        trial_n = len(trial.study.trials)
        best_score = (
            score
            if trial_n == 1 or score > trial.study.best_value
            else trial.study.best_value
        )

        logger.info(
            "Trials: %s, Best Score: %s, Score %s", trial_n, best_score, score
        )
        logger.debug("Parameters: %s", str(param))
        return score

    @property
    def coef_(self):
        check_is_fitted(self)

        if not self.refit:
            raise AttributeError("refit must be true to use property")

        # If we have a pipeline, we need to access the estimator from it
        return (
            self.best_estimator_[self.pipe_estimator].coef_
            if self.pipe_estimator
            else self.best_estimator_.coef_
        )

    @property
    def feature_importances_(self):
        check_is_fitted(self)

        if not self.refit:
            raise AttributeError("refit must be true to use property")

        # If we have a pipeline, we need to access the estimator from it
        return (
            self.best_estimator_[self.pipe_estimator].feature_importances_
            if self.pipe_estimator
            else self.best_estimator_.feature_importances_
        )

    @property
    def get_study_(self):
        check_is_fitted(self)
        return self.study_

    @property
    def best_score_(self):
        check_is_fitted(self)
        return self.study_.best_value

    @property
    def best_params_(self):
        check_is_fitted(self)
        return self.study_.best_params
