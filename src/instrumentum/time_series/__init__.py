
    
def ts_print(df, col_target, col_key, value_true=1):
    """ Print a Pandas dataframe with the required format

    Parameters
    ----------
    df : Dataframe
        A dataframe in the required timeframe form
    col_target : [type]
        [description]
    col_key : [type]
        [description]
    value_true : int, optional
        [description], by default 1
    """
    
    
    # Let's get all the customers that had at least one open card (drop duplicates; same cust can appear more than once)
    keys_pos_unique = df.loc[df[col_target] == value_true][col_key].drop_duplicates(
        keep="first"
    )

    # Let's get all customers who neer had a card opened (Since cust can be in many datasets, drop duplicates)
    keys_neg_unique = df.loc[~df[col_key].isin(keys_pos_unique)][
        col_key
    ].drop_duplicates(keep="first")

    print("Total unique keys that with a possitive value: ", len(keys_pos_unique))
    print(
        "Total records of ckeys that with a possitive value: ",
        df.loc[df[col_key].isin(keys_pos_unique)].shape[0],
    )

    print("Total unique keys with no possitive value: ", len(keys_neg_unique))
    print(
        "Total records of customers with no possitive values: ",
        df.loc[df[col_key].isin(keys_neg_unique)].shape[0],
    )
