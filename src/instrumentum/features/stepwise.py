from itertools import combinations 
from itertools import chain

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
import logging

import time

from joblib import Parallel, delayed
import multiprocessing

logger = logging.getLogger(__name__)
    
def _default_scoring(X_train, y_train):
    
    model = DecisionTreeClassifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    return cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv).mean()
      
    
def _get_combs(set_size, combs, include_empty=False):

    l_comb = [combinations(list(range(0, set_size)), x) \
        for x in range((0 if include_empty else 1), combs + 1)]

    return chain.from_iterable(l_comb)
   


# TODO: en vez de drop real y las condiciones, tener una variable que vaya agregando los globales que se eliminan
def backward_stepwise(X_train, y_train, n_combs=1, rounding=4, remove_always=False, _scorer = None):

    scorer = _default_scoring
    
    if(_scorer is not None):
        if(not hasattr(_scorer, '__call__')):
            raise ValueError("Value provided for scorer is not a callable function")
        
        scorer = _scorer
        
    print("Number of combinations: ", n_combs)
    print("Training shape: ", X_train.shape)
    print("Label distribution: \n", y_train.value_counts())
    
    X_train = X_train.copy()

    result_global = round(scorer(X_train, y_train), rounding)
  
    print("\nInitial scoring with all columns: ", result_global)
    while True:

        columns_to_remove   = [None]
        best_result_local   = 0

        combs = list(_get_combs(len(X_train.columns),n_combs))
        combs.pop(0) # remove the empty set
        
        print("Combinations to test: {}".format(len(combs)))
        for comb in combs:
            l_comb = list(comb)
            
            result_local = round(scorer(X_train.drop(X_train.columns[l_comb],axis=1), y_train), rounding)
                
            if result_local > best_result_local:
                best_result_local = result_local
                columns_to_remove = l_comb


        # equal is important below, so all being equal, keep moving and removing columns
        if (best_result_local >= result_global or remove_always) and (len(X_train.columns)>1):
            print("Best score: {}, previous {}, columns removed: {}".format(best_result_local, result_global, list(X_train.columns[columns_to_remove])))
            print("Best columns so far: {}".format(list(X_train.drop(X_train.columns[columns_to_remove],axis=1).columns)))
            result_global = best_result_local

            X_train.drop(X_train.columns[columns_to_remove],axis=1, inplace=True)
              
        else:
            print("\nBest score: {}, columns final: {}".format(result_global, list(X_train.columns)))
            break


        
def _run_scorer(X_train, y_train, rounding, tracker_cols, comb, scorer, verbose):
    cols_not_yet_added = [c for c in X_train.columns if c not in tracker_cols]
    cols_comb = list(X_train[cols_not_yet_added].columns[comb])
    cols_to_test =  tracker_cols + cols_comb
    score = scorer(X_train[cols_to_test], y_train)
    score = round(score, rounding)
    
    logger.setLevel(verbose)
    logger.debug("Score %s for this combination: %s", score, cols_comb)
    
    return score, cols_comb

# TODO, return a tuple or two lists with the columns and the aggregated score
def forward_stepwise(X_train, y_train, n_combs=1, rounding=4, add_always=False, _scorer = None, verbose=logging.INFO, n_jobs=-1):


    logger.setLevel(verbose)
    
    scorer = _default_scoring
    
    if(_scorer is not None):
        if(not hasattr(_scorer, '__call__')):
            raise ValueError("Value provided for scorer is not a callable function")
        
        scorer = _scorer
    
    max_jobs = multiprocessing.cpu_count()
    if(n_jobs != -1):
        if(n_jobs > max_jobs):
            logger.warning("Max processors in this coputer %s, lowering to that from input %s", max_jobs, n_jobs)
            n_jobs = max_jobs

    logger.info("Number of cores to be used: %s\n (-1 = all), total cores: %s\n", n_jobs, max_jobs)
    keep_going = True
    
    all_cols = list(X_train.columns)
    tracker_cols  = []
    tracker_score = 0
    
    start_time = time.time()
    
    while keep_going:
       
        n_cols_remaining = len(all_cols) - len(tracker_cols)
        combs = list(_get_combs(n_cols_remaining,n_combs))
        
        logger.info("Remaining columns to test: %s", n_cols_remaining)
        logger.info("Combinations to test: %s",len(combs))

        ret = Parallel(n_jobs=n_jobs) \
            (delayed(_run_scorer) \
                (X_train, y_train, rounding, tracker_cols, list(comb), scorer, verbose) \
                    for comb in combs)

        best_comb_score, best_comb_cols = max(ret,key=lambda item:item[0])
        
        logger.info("Best score from combinations: %s, global score %s", best_comb_score, tracker_score)
        logger.info("Best score comes from columns: %s", best_comb_cols)
        
        if (best_comb_score > tracker_score or add_always):
            
            tracker_cols += best_comb_cols
            tracker_score = best_comb_score
            
            logger.info("Best columns were added. All columns added so far %s\n", tracker_cols)
                
        else:
            logger.info("Columns were not added as they do not increase score. Finishing\n")
            keep_going = False

        if(len(tracker_cols) == len(all_cols)):
            logger.info("All columns were added. Finishing\n")
            keep_going = False
    
    logger.info("Total time: %s seconds" % (time.time() - start_time))
    
    return tracker_cols