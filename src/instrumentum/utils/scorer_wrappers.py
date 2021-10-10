from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import optuna


class Objective_xgb(object):
    def __init__(self, X_train, y_train, X_test, y_test, n_repeats=3, verbose=True):

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.score = 0
        self.counter = 0

        self.n_repeats = n_repeats
        self.verbose = verbose

    def __call__(self, trial):

        param = {
            "objective": "binary:logistic",
            # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 1e-8, 1.0),
            "subsample": trial.suggest_loguniform("subsample", 1e-8, 1.0),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 150),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
            "subsample": trial.suggest_loguniform("subsample", 1e-8, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 5, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 9),
            "gamma": trial.suggest_loguniform("gamma", 1e-8, 1e1),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-8, 1e1),
            #   "eta" : trial.suggest_loguniform("eta", 1e-8, 1.0),
        }

        #         if param["booster"] == "gbtree" or param["booster"] == "dart":
        #             param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        #             param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        #             param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1e1)
        #             param["min_child_weight"] = trial.suggest_loguniform("min_child_weight", 1e-8, 1e1)
        #             param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        #         if param["booster"] == "dart":
        #             param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        #             param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        #             param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        #             param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        estimator = XGBClassifier(**param, eval_metric="auc", use_label_encoder=False)

        cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=self.n_repeats)

        #      custom_scorer = make_scorer(my_scorer, needs_proba=True)

        score = cross_val_score(
            estimator, X=self.X_train, y=self.y_train, cv=cv, scoring="roc_auc"
        ).mean()

        if self.X_test is not None and self.y_test is not None:

            estimator.fit(self.X_train, self.y_train)
            proba = estimator.predict_proba(self.X_test)[:, 1]
            score = roc_auc_score(self.y_test, proba)

        if self.verbose == True:
            print("Iteration: ", self.counter, " Score: ", score)
        self.counter += 1

        if score > self.score:
            self.score = score
            # print(score)

        return score


class Objective_rf(object):
    def __init__(self, X_train, y_train, X_test, y_test, n_repeats=3, verbose=True):

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.score = 0
        self.counter = 0

        self.n_repeats = n_repeats
        self.verbose = verbose

    def __call__(self, trial):

        param = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 120),
            "max_depth": int(trial.suggest_loguniform("max_depth", 1, 14)),
        }

        estimator = RandomForestClassifier(**param)

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=self.n_repeats)

        score = cross_val_score(
            estimator, X=self.X_train, y=self.y_train, cv=cv, scoring="roc_auc"
        ).mean()

        if self.X_test is not None and self.y_test is not None:

            estimator.fit(self.X_train, self.y_train)
            proba = estimator.predict_proba(self.X_test)[:, 1]
            score = roc_auc_score(self.y_test, proba)

        if self.verbose == True:
            print("Iteration: ", self.counter, " Score: ", score)
        self.counter += 1

        if score > self.score:
            self.score = score
            # print(score)

        return score


def optimizer_optuna_rf(
    X_train, y_train, X_test=None, y_test=None, n_trials=15, n_repeats=2, verbose=False
):

    objective = Objective_rf(X_train, y_train, X_test, y_test, n_repeats, verbose)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials)

    rf = RandomForestClassifier(**study.best_trial.params)

    return study.best_trial.value, rf


def optimizer_optuna_xgb(
    X_train, y_train, X_test=None, y_test=None, n_trials=250, n_repeats=3, verbose=False
):

    objective = Objective_xgb(X_train, y_train, X_test, y_test, n_repeats, verbose)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_xgb = optuna.create_study(direction="maximize")

    study_xgb.optimize(objective, n_trials)

    # print(
    #     "Best trial: score {}, params {}".format(
    #         study_xgb.best_trial.value, study_xgb.best_trial.params
    #     )
    # )

    xgb_optuna = XGBClassifier(**study_xgb.best_trial.params)
    xgb_optuna.fit(X_train, y_train)

    print("Best features:")

    a = pd.DataFrame(xgb_optuna.get_booster().feature_names)
    a["b"] = xgb_optuna.feature_importances_.tolist()
    a.sort_values(by="b", ascending=False, inplace=True)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(a)

    return study_xgb.best_trial.value, xgb_optuna
