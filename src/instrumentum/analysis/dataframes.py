import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_col_nans(df, cutoff=0):
    # nan_rows = df[df.isnull().T.any()]
    perc = df.isnull().mean()

    return perc.loc[perc > cutoff]


def get_col_non_numeric(df):
    return df.select_dtypes(exclude=[np.number]).dtypes
