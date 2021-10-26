from itertools import combinations
from itertools import chain

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
import logging

import time

from joblib import Parallel, delayed
import multiprocessing

from instrumentum.utils.decorators import timeit

logger = logging.getLogger(__name__)


def _default_scoring(X_train, y_train):

    model = DecisionTreeClassifier()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)

    return cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv).mean()


def _get_combs(set_size, combs, include_empty=False):

    l_comb = [
        combinations(list(range(0, set_size)), x)
        for x in range((0 if include_empty else 1), combs + 1)
    ]

    return chain.from_iterable(l_comb)


# TODO: en vez de drop real y las condiciones, tener una variable que vaya agregando los globales que se eliminan
def backward_stepwise(
    X_train, y_train, n_combs=1, rounding=4, remove_always=False, _scorer=None
):

    scorer = _default_scoring

    if _scorer is not None:
        if not hasattr(_scorer, "__call__"):
            raise ValueError("Value provided for scorer is not a callable function")

        scorer = _scorer

    print("Number of combinations: ", n_combs)
    print("Training shape: ", X_train.shape)
    print("Label distribution: \n", y_train.value_counts())

    X_train = X_train.copy()

    result_global = round(scorer(X_train, y_train), rounding)

    print("\nInitial scoring with all columns: ", result_global)
    while True:

        columns_to_remove = [None]
        best_result_local = 0

        combs = list(_get_combs(len(X_train.columns), n_combs))
        combs.pop(0)  # remove the empty set

        print("Combinations to test: {}".format(len(combs)))
        for comb in combs:
            l_comb = list(comb)

            result_local = round(
                scorer(X_train.drop(X_train.columns[l_comb], axis=1), y_train), rounding
            )

            if result_local > best_result_local:
                best_result_local = result_local
                columns_to_remove = l_comb

        # equal is important below, so all being equal, keep moving and removing columns
        if (best_result_local >= result_global or remove_always) and (
            len(X_train.columns) > 1
        ):
            print(
                "Best score: {}, previous {}, columns removed: {}".format(
                    best_result_local,
                    result_global,
                    list(X_train.columns[columns_to_remove]),
                )
            )
            print(
                "Best columns so far: {}".format(
                    list(
                        X_train.drop(X_train.columns[columns_to_remove], axis=1).columns
                    )
                )
            )
            result_global = best_result_local

            X_train.drop(X_train.columns[columns_to_remove], axis=1, inplace=True)

        else:
            print(
                "\nBest score: {}, columns final: {}".format(
                    result_global, list(X_train.columns)
                )
            )
            break


def _run_scorer(X_train, y_train, rounding, tracker_cols, comb, scorer, verbose):
    cols_not_yet_added = [c for c in X_train.columns if c not in tracker_cols]
    cols_comb = [cols_not_yet_added[idx] for idx in comb]

    cols_to_test = tracker_cols + cols_comb

    score = scorer(X_train[cols_to_test], y_train)
    score = round(score, rounding)

    logger.setLevel(verbose)
    logger.debug("Score %s for this combination: %s", score, cols_comb)

    return score, cols_comb


@timeit
def forward_stepwise(
    X,
    y,
    n_combs=1,
    rounding=4,
    add_always=False,
    scorer=None,
    verbose=logging.INFO,
    n_jobs=-1,
):
    """
    Function that tries to find the columns that offer the best prediction power,
    based on the training and label information provided.
    The algorithm of forward stepwise is well known. This function expands on the
    concept by trying to select more than 1 column at each iteration. The number of
    columns could be up to the parameter n_combs. One could try to search over all
    the possible combinations of pending columns but the resources required tend to
    infinite. The parameter n_combs control how many combinations to test based on
    the columns not yeat added

    Parameters
    ----------
    X : DataFrame
        Training information
    y : Series
        Label results of the training information
    n_combs : int, optional
        Number of combinations to test from the remaining columns. If set to 1, the
        function adds at most 1 column at each iteration, and behaves as the standard
        forward stepwise algorithm, by default 1
    rounding : int, optional
        The function (if add_always is False) will try to add new columns only if they
        improve the performance (in theory additional columns would never make the metric worse). 
        We might not be interested in keep adding columns if the performance is slightly improved,
        or none at all. This parameter rounds the result of the scoring such that the smaller this value,
        the less likely it will keep adding columns. For example, if adding column x adds
        0.001 of prediction power, but rounding is 2, this will be 0 and the iteration will
        not add it, by default 4
    add_always : bool, optional
        If one needs to keep adding the best columns, regardless if they improve the overall
        performance, this variable must be True. If true, the function will return all the
        columns of the training dataset, sorted by the scoring produced for their additions,
        by default False
    scorer : [type], optional
        [description], by default None
    verbose : [type], optional
        [description], by default logging.INFO
    n_jobs : int, optional
        [description], by default -1

    Returns
    -------
    list
        A list of tuples (ordered) of the columns selected

    """
    logger.setLevel(verbose)

    if scorer is not None:
        if not hasattr(scorer, "__call__"):
            raise ValueError("Value provided for scorer is not a callable function")
    else:
        scorer = _default_scoring

    max_jobs = multiprocessing.cpu_count()
    if n_jobs != -1:
        if n_jobs > max_jobs:
            logger.warning(
                "Max cores in this coputer %s, lowering to that from input %s",
                max_jobs,
                n_jobs,
            )
            n_jobs = max_jobs

    logger.info(
        "Number of cores to be used: %s, total available: %s\n",
        max_jobs if n_jobs == -1 else n_jobs,
        max_jobs,
    )

    tracker_cols, tracker_score = [], 0
    return_data = []

    keep_going = True

    while keep_going:

        n_cols_remaining = len(X.columns) - len(tracker_cols)
        combs = list(_get_combs(n_cols_remaining, n_combs))

        logger.info("Remaining columns to test: %s", n_cols_remaining)
        logger.info("Combinations to test: %s", len(combs))

        comb_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_scorer)(
                X,
                y,
                rounding,
                tracker_cols,
                list(comb),
                scorer,
                verbose,
            )
            for comb in combs
        )

        best_comb_score, best_comb_cols = max(comb_results, key=lambda item: item[0])

        logger.info(
            "Best score from combinations: %s, global score %s",
            best_comb_score,
            tracker_score,
        )
        logger.info("Best score comes from columns: %s", best_comb_cols)

        # Check if finish conditions are met
        if best_comb_score > tracker_score or add_always:

            tracker_cols += best_comb_cols
            tracker_score = best_comb_score

            logger.info(
                "Best columns were added. All columns added so far %s\n", tracker_cols
            )

        else:
            logger.info(
                "Columns were not added as they do not increase score. Finishing\n"
            )
            keep_going = False

        if len(tracker_cols) == len(X.columns):
            logger.info("All columns were added. Finishing\n")
            keep_going = False

        return_data.append((best_comb_cols, best_comb_score))

    return return_data
