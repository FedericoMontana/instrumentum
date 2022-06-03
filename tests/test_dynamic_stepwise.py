import logging

import pytest

from instrumentum.feature_selection.stepwise import DynamicStepwise
from instrumentum.model_tuning._optuna_dispatchers import optuna_param_disp
from instrumentum.model_tuning.wrapper_optuna import OptunaSearchCV
from tests.conftest import ESTIMATORS


def get_dynamicstepwise(X, y, estimator, combs=1, rounding=3, **kwars):

    search_function = optuna_param_disp[estimator.__class__.__name__]

    os = OptunaSearchCV(
        estimator=estimator,
        scoring="roc_auc",
        cv=5,
        search_space=search_function,
        n_iter=5,
        random_state=1,
        verbose=logging.ERROR,
    )

    stepw = DynamicStepwise(
        estimator=os,
        rounding=rounding,
        n_combs=combs,
        verbose=logging.ERROR,
        direction="forward",
        **kwars,
    )
    stepw.fit(X, y)

    return stepw


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_single_combo_fwd(frame_01, estimator):

    X, y = frame_01
    stepw = get_dynamicstepwise(X, y, combs=1, rounding=3, estimator=estimator)

    print(stepw.seq_columns_selected_)

    # I use 6 in case it picks randomnly an additional one
    assert len(stepw.seq_columns_selected_) <= 6

    assert stepw.seq_columns_selected_[0][1][0] == "x0"
    assert stepw.seq_columns_selected_[1][1][0] == "x1"
    assert stepw.seq_columns_selected_[2][1][0] == "x2"
    assert stepw.seq_columns_selected_[3][1][0] == "x3"
    assert stepw.seq_columns_selected_[4][1][0] == "x4"


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_multiple_combo_fwd(frame_01, estimator):

    X, y = frame_01
    stepw = get_dynamicstepwise(X, y, combs=2, rounding=3, estimator=estimator)

    # This len should be always 3, 4 just in case some noise
    # is picked up
    assert len(stepw.seq_columns_selected_) <= 4
    assert set(stepw.seq_columns_selected_[0][1]) == set(["x0", "x1"])
    assert set(stepw.seq_columns_selected_[1][1]) == set(["x2", "x3"])
    assert "x4" in set(stepw.seq_columns_selected_[2][1])


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_max_cols(frame_01, estimator):

    X, y = frame_01

    # Test 1: max_cols only
    kwars = {"max_cols": 3}
    stepw = get_dynamicstepwise(
        X, y, combs=2, rounding=3, estimator=estimator, **kwars
    )

    # If ask for 3, these 3 should always be included
    assert set(stepw.get_feature_names_out()) == set(["x0", "x1", "x2"])

    # Test 2: max_cols must not return exactly the number provided, it is
    # "up to"
    kwars = {"max_cols": 9}
    stepw = get_dynamicstepwise(
        X, y, combs=2, rounding=3, estimator=estimator, **kwars
    )
    feat = stepw.get_feature_names_out()

    # Only 5 are goods. This should be always true, and be ==5 ideally
    assert len(feat) < 7

    # These should always be included
    assert set(["x0", "x1", "x2", "x3", "x4"]).issubset(feat)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_add_always(frame_01, estimator):

    X, y = frame_01

    kwars = {"max_cols": 9, "add_always": True}
    stepw = get_dynamicstepwise(
        X, y, combs=2, rounding=3, estimator=estimator, **kwars
    )
    feat = stepw.get_feature_names_out()
    assert (
        len(feat) == 9
    )  # because of add_always, max_cols should match the total

    # These should always be included
    assert set(["x0", "x1", "x2", "x3", "x4"]).issubset(feat)
