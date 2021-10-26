import logging
import time

logger = logging.getLogger(__name__)

def timeit(func):
    def wrap_func(*args, **kwargs):
       
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
  
        if 'verbose' in kwargs:
            verbose = kwargs.get('verbose', wrap_func.__name__.upper())
            logger.setLevel(verbose)
        
        logger.info('Function %s executed in %s seconds', func.__name__, (t2-t1))
        return result
    
    return wrap_func
