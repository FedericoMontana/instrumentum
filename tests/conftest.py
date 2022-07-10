import itertools

import lightgbm as ltb
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from instrumentum.utils.utils import get_random_mask

# These will be used in the paramterized fixture within each test file
ESTIMATORS = [DecisionTreeClassifier(), LogisticRegression()]
COMBS = [1, 2]

ESTIMATORS_CLUSTERS = [DecisionTreeRegressor(), ltb.LGBMRegressor()]


def get_params():
    for x in itertools.product(ESTIMATORS, COMBS):
        yield x


@pytest.fixture
def frame_01(scope="package"):

    np.random.seed(0)

    n_r, n_c = 10000, 11

    # [[column, similarity with y], ... ]
    predictaibility = [[0, 0.3], [1, 0.25], [2, 0.20], [3, 0.15], [4, 0.1]]

    X = np.random.randint(0, 2, size=(n_r, n_c))
    y = np.random.randint(0, 2, size=(n_r))

    for c, perc in predictaibility:

        mask = get_random_mask(perc, n_r)
        X[mask, c] = y[mask]

    # Numpy to Pandas
    X = pd.DataFrame(X, columns=["x" + str(x) for x in range(n_c)])
    # Shuffle the columns, to make it more random
    X = X[np.random.permutation(X.columns)]
    return X, y


@pytest.fixture
def frame_02(scope="package"):
    # Generates 9 variables. correlated in sets of three

    np.random.seed(0)

    size = 50000

    normal = np.random.normal(0, 1, size)
    expo = np.random.exponential(1.0, size)
    gamma = np.random.gamma(0.1, 1, size)

    dists = np.vstack(
        (
            (
                normal,
                normal + np.random.normal(0, 2, size),
                normal + np.random.normal(0, 2, size),
                expo,
                expo + np.random.normal(0, 2, size),
                expo + np.random.normal(0, 2, size),
                gamma,
                gamma + np.random.normal(0, 2, size),
                gamma + np.random.normal(0, 2, size),
            )
        )
    ).T

    y = np.random.uniform(-1, 1, size)

    # For these columns, lets create a relationship 20% with target
    for c in [0, 3, 6]:
        mask = get_random_mask(0.2, size)
        dists[mask, c] = y[mask]

    X = pd.DataFrame(
        dists, columns=["x" + str(x) for x in range(dists.shape[1])]
    )
    X = X[np.random.permutation(X.columns)]

    return X, y
