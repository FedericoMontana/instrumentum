import logging

import pandas as pd

logger = logging.getLogger(__name__)

def trasform_to_numeric(df, verbose=logging.INFO):
    
    logger.setLevel(verbose)
    
    for col in df.columns:
        logger.info("\nProcessing: %s", col)
        
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        logger.info("is_numeric")

def remove_unneded_columns():
    
    # remove constants, etc
    pass