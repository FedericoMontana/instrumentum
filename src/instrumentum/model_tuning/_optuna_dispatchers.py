import logging

import optuna
import optuna.integration.lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor


def _xgbclassifier_default(trial: optuna.trial.Trial):
    param = {

        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 20, 600, step=20),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    return param


def _lgbmclassifier_default(trial: optuna.trial.Trial):
    params = {
        #"verbosity": -1,
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["gbdt", "dart", "goss"]
        ),
        "verbose": -1,
        "objective": "binary",
        "metric": ["binary", "binary_error", "auc"],
        "num_leaves": trial.suggest_int("num_leaves", 5, 500, step=5),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.0, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 30),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 30),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
    }
    # if params["boosting_type"] == "dart":
    #     params["drop_rate"] = trial.suggest_loguniform("drop_rate", 1e-8, 1.0)
    #     params["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)
    # if params["boosting_type"] == "goss":
    #     params["top_rate"] = trial.suggest_uniform("top_rate", 0.0, 1.0)
    #     params["other_rate"] = trial.suggest_uniform(
    #         "other_rate", 0.0, 1.0 - params["top_rate"]
    #     )

    return params


def _catboostclassifier_default(trial: optuna.trial.Trial):
    params = {
        "allow_writing_files": False,
        "logging_level": 'Silent',
        #"silent": True,
        #"verbose": 0,
        "iterations": trial.suggest_int("iterations", 50, 300),
        "depth": trial.suggest_int("depth", 2, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "random_strength": trial.suggest_int("random_strength", 0, 100),
        "bagging_temperature": trial.suggest_loguniform(
            "bagging_temperature", 0.01, 100.00
        ),
        "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        "od_wait": trial.suggest_int("od_wait", 10, 50),
    }

    return params


def _random_forest_classifier_default(trial: optuna.trial.Trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 120),
        "max_depth": trial.suggest_int("max_depth", 1, 12)
    }

    return params

def _decision_tree_classifier_default(trial: optuna.trial.Trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
    }

    return params


def _decision_tree_regressor_default(trial: optuna.trial.Trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 12),
      #  "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
    }

    return params

optuna_param_disp = {
    XGBRegressor.__name__: _xgbclassifier_default,
    LGBMClassifier.__name__: _lgbmclassifier_default,
    RandomForestClassifier.__name__: _random_forest_classifier_default,
    CatBoostClassifier.__name__: _catboostclassifier_default,
    DecisionTreeClassifier.__name__: _decision_tree_classifier_default,
    DecisionTreeRegressor.__name__: _decision_tree_regressor_default,
}
