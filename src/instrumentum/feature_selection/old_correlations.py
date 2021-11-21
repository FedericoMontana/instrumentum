
import numpy as np
import pandas as pd

from instrumentum.utils._decorators import timeit


def _min_corr_groups(df: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
   
    corr = df.corr().abs()
    corr.loc[:,:] =  np.tril(corr, k=-1)
    corr = corr.unstack()
    corr = corr[corr >= threshold]
    corr.sort_values(inplace=True, ascending=True)
    return [set(x) for x in corr.index]

def _can_add_item(item, group, corr_pairs):

    if item in group:
        return False

    all_combs = [{item,x} for x in group]

    if all(any(y.issubset(x) for x in corr_pairs) for y in all_combs):
        return True

def _remove_redundancy(l):
    # this line removes the sets that are equal
    l = [set(item) for item in set(frozenset(item) for item in l)]

    # this removes all elements included in other (if I dont execute the previous line)
    # then sets that are equal are both removed
    return [x for x in l if not any(x<=y for y in l if x is not y)]


def _get_correlated_items(item, corr_pairs):
    
    return set().union(*[x for x in corr_pairs if item in x])



def get_corr_groups_accurate(df, threshold):

    corr_pairs = _min_corr_groups(df, threshold)

    # The initial groups will be the minimum expression
    corr_groups = corr_pairs.copy()

    # These are the columns we need, let's avoid to process the ones
    # that do not have any pair over the threshold
    columns = set().union(*corr_pairs)

    keep_going = True
    while(keep_going):

        new_groups = []
        for group in corr_groups:
            for col in columns:
                # If the item can be added to an existing group, then I will not add
                # it directly to that specific groupo, but rather will create another entry with them
                # the reason is that another variable can come in later and do not fit in becuase of the 
                # recently added item, thus we will leave the original group for every variable to try.
                # later we will remove redundancy.
                if _can_add_item(col, group, corr_pairs):
                    new_groups.append(group | set([col]))

        if(len(new_groups)!=0):
            corr_groups += new_groups           
            corr_groups = _remove_redundancy(corr_groups)
        else:
            keep_going = False

    return corr_groups


def get_corr_simple(df: pd.DataFrame, threshold: float):
    # This is not deterministic

    # Get list of all pairs correlated (based on threshold)
    corr_pairs = _min_corr_groups(df, threshold)
    
    corr_groups = []

    # These are the columns we need, let's avoid to process the ones
    # that do not have any pair correlated
    columns = set().union(*corr_pairs)

    for col in columns:
        added_items = set().union(*corr_groups)

        if(col not in added_items):
            correlated_items = _get_correlated_items(col, corr_pairs)

            correlated_items.difference_update(added_items)

            if(len(correlated_items)>1): # 0 or itself no append
                corr_groups.append(correlated_items)

        
    return corr_groups
