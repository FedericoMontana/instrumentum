import logging

import optuna
import optuna.integration.lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor, plot_importance

from instrumentum.model_tuning._optuna_dispatchers import optuna_param_disp

logger = logging.getLogger(__name__)


def _opt_generic_objective(X, y, trial, estimator, cv, metric):

    param = optuna_param_disp[estimator.__name__](trial)
    estimator = estimator(**param)

    score = cross_val_score(estimator, X=X, y=y, cv=cv, scoring=metric).mean()

    trial_n = len(trial.study.trials)
    best_score = (
        score
        if trial_n == 1 or score > trial.study.best_value
        else trial.study.best_value
    )

    logger.info("Trials: %s, Best Score: %s, Score %s", trial_n, best_score, score)
    return score


def wrapper_opt(
    X,
    y,
    estimator=None,
    metric="roc_auc",
    n_trials=5,
    verbose=logging.INFO,
    return_fit=True,
    direction="maximize",
    cv_splits=5,
    cv_repeats=1,
):
    # Our Logger
    logger.setLevel(verbose)
    # Let's turn off the verbosity of optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats)
    estimator = estimator or DecisionTreeClassifier

    logger.info("Estimator received: %s, trials: %s\n", estimator.__name__, n_trials)

    study = optuna.create_study(direction=direction)
    study.optimize(
        lambda trial: _opt_generic_objective(
            trial=trial,
            X=X,
            y=y,
            estimator=estimator,
            cv=cv,
            metric=metric,
        ),
        n_trials=n_trials,
    )

    estimator = estimator(**study.best_params)
    return_fit and estimator.fit(X, y)

    return study.best_trial.value, estimator


def wrapper_opt_lgbm(
    X, y, metric="auc", time_budget=120, verbose=logging.INFO, return_fit=False
):

    # Our Logger
    logger.setLevel(verbose)

    # Let's turn off the verbosity of optuna and lighgbm
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    no_logger = logging.getLogger("sd")
    no_logger.addHandler(logging.NullHandler())
    lgb.register_logger(no_logger)

    def log_trials(std, frz_trial):
        logger.info(
            "\nTrials: %s, Iteration Score: %s", len(std.trials), std.best_value
        )

    params = {
        "objective": "binary",
        "metric": metric,
        "boosting_type": "gbdt",
        "seed": 42,
    }

    dtrain = lgb.Dataset(X, label=y)
    rkf = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=2,
        random_state=42,
    )
    study_tuner = optuna.create_study(direction="maximize")

    tuner = lgb.LightGBMTunerCV(
        params,
        dtrain,
        study=study_tuner,
        time_budget=time_budget,
        seed=42,
        optuna_callbacks=[log_trials],
        show_progress_bar=False,
        folds=rkf,
    )

    tuner.run()

    lgbm = LGBMClassifier(**tuner.best_params)
    return_fit and lgbm.fit(X, y)

    return tuner.best_score, lgbm
