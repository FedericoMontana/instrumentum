import logging

import pytest

from instrumentum.feature_selection.correlation import ClusterSelection
from instrumentum.feature_selection.stepwise import DynamicStepwise
from instrumentum.model_tuning._optuna_dispatchers import optuna_param_disp
from instrumentum.model_tuning.wrapper_optuna import OptunaSearchCV
from tests.conftest import ESTIMATORS_CLUSTERS


def get_clusterclf(X, y, estimator):

    search_function = optuna_param_disp[estimator.__class__.__name__]

    os = OptunaSearchCV(
        estimator=estimator,
        scoring="neg_mean_squared_error",
        cv=5,
        search_space=search_function,
        n_iter=5,
        random_state=1,
        verbose=logging.WARNING,
    )

    stepw = DynamicStepwise(
        estimator=os,
        rounding=2,
        n_combs=1,
        max_cols=1,
        verbose=logging.WARNING,
    )

    clsl = ClusterSelection(
        t=3,
        criterion="maxclust",
        meta_estimator=stepw,
        verbose=logging.WARNING,
    )
    clsl.fit(X, y)

    return clsl


@pytest.mark.parametrize("estimator", ESTIMATORS_CLUSTERS)
def test_cluster_selection(frame_02, estimator):
    # test that is creates the clusters  and selects the best
    # columns withim them
    # columns x0, x3 and x6 have correlation with the target, the other
    # do not
    X, y = frame_02

    clsl = get_clusterclf(X, y, estimator)

    assert set(clsl.get_feature_names_out()) == set(["x0", "x3", "x6"])


@pytest.mark.parametrize("estimator", ESTIMATORS_CLUSTERS)
def test_cluster_generation(frame_02, estimator):
    # test that confirms that the test are created correctly
    X, y = frame_02

    clsl = get_clusterclf(X, y, estimator)

    def set_ors(x):
        s = set(x)
        return (
            s == set(["x0", "x1", "x2"])
            or s == set(["x3", "x4", "x5"])
            or s == set(["x6", "x7", "x8"])
        )

    # I know the sets, but don't know the order they come in

    f = clsl.feature_names_in_
    assert set_ors(f.take(clsl.clusters_in_[1]))
    assert set_ors(f.take(clsl.clusters_in_[2]))
    assert set_ors(f.take(clsl.clusters_in_[3]))
