import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from instrumentum.utils.utils import get_random_mask

# from lightgbm import LGBMClassifier
# from sklearn.linear_model import LogisticRegression
# These will be used in the paramterized fixture within each test file
ESTIMATORS = [DecisionTreeClassifier()]  # , LogisticRegression()]


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
