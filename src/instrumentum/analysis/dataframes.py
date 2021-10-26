import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_col_nans(df, cutoff=0):
    #nan_rows = df[df.isnull().T.any()]
    perc = data_df.isnull().mean() 

    return perc.loc[perc > cutoff]

def get_col_non_numeric(df):
    return df.select_dtypes(exclude=[np.number]).dtypes
    


import numpy as np
if __name__ == "__main__":
    
    raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'],\
        'last_name': [pd.Timestamp('20180310'), pd.Timestamp('20180310'), pd.Timestamp('20180310'), pd.Timestamp('20180310'), pd.Timestamp('20180310')], \
            'age': [22, 2, 2, 24, 25], \
                'sex': ['m', np.nan, 'f', 'm', 'f'], \
                    'Test1_Score': [False, True, "la", True, True],\
                        'Test2_Score': [25, np.nan, np.nan, 0, 0]}
    
    data_df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'sex', 'Test1_Score', 'Test2_Score'])

    a = get_col_nans(data_df)
    b = get_col_non_numeric(data_df)
    
    print(pd.concat([a, b], axis=1, join="outer"))