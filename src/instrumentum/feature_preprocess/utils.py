# General utilities for pre-processing

from dateutil.parser import parse
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

import numpy as np


def remove_shit(dataframe, keep=None):

    df = dataframe.copy()

    # remove non numeric
    non_numeric = list(df.select_dtypes(exclude=np.number).columns)
    print("Removing non-numeric columns: ", non_numeric)
    df = df.select_dtypes(np.number)
    df.replace([np.inf], 999999, inplace=True)
    df.replace([-np.inf], -999999, inplace=True)

    almost_constant = list(
        df.columns[~VarianceThreshold(threshold=0.001).fit(df).get_support()]
    )
    all_distinct = list(df.columns[df.nunique() == len(df)])

    print("Almost-constant features: ", almost_constant)
    print("All-distinct-value features: ", all_distinct)

    to_remove = almost_constant + all_distinct
    to_remove = [i for i in to_remove if i not in (keep or [])]
    return df.drop(to_remove, axis=1) if to_remove else df


################################


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    string = str(string)

    # Some checks
    if len(string) < 4 or len(string) == 5:
        return False

    try:
        if int(string) <= 0:
            return False
    except ValueError:
        pass
    ###############################

    try:
        ret = parse(string, fuzzy=fuzzy)
        year = int(ret.strftime("%Y"))

        if year > 2025 or year < 1940:
            return False

        return True

    except:
        return False


# returns 0 if NAN or similar, otherwise returns year
def get_number_from_date(string, fuzzy=False):

    ERR_RET = 0
    string = str(string)
    if is_date(string, True):
        ret = parse(string, fuzzy=fuzzy)
        return int(ret.strftime("%Y%m"))

    else:
        return ERR_RET


def rescue_non_numeric(df_, nan_code=None):

    df = df_.copy()
    non_numeric = df.select_dtypes(exclude=np.number).columns.tolist()

    for col in non_numeric:
        n_rows = len(df[df[col] != nan_code])
        print("Evaluating: ", col, " % non-nan: ", n_rows / len(df))

        if df[col].apply(lambda x: is_date(x, True)).sum() > int(n_rows * 0.8):
            print("-- Converted to Date")
            df[col] = df[col].apply(lambda x: get_number_from_date(x, True))
        elif df[col].nunique() < 30:
            print("--  Label Encoded")
            lb_style = preprocessing.LabelEncoder()
            df[col] = lb_style.fit_transform(df[col].astype("str")).astype("int64")
        else:
            print("-- Remains as object")

    return df
