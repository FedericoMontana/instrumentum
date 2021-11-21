import logging
import multiprocessing

logger = logging.getLogger(__name__)

def check_jobs(n_jobs, verbose=logging.INFO):
    
    logger.setLevel(verbose)
    
    verified_jobs = n_jobs
    max_jobs = multiprocessing.cpu_count()
 
    if n_jobs > max_jobs:
        logger.warning(
            "Max cores in this coputer %s, lowering to that from input %s",
            max_jobs,
            n_jobs,
        )
        verified_jobs = max_jobs
    elif n_jobs < -1:
        logger.error("Jobs cannot be negative (except for -1 for all cores). Your total cores \
            are %s , modifying your entry to 1", max_jobs)
        verified_jobs = 1
        
    else:
        if(n_jobs==-1):
            verified_jobs = max_jobs
            
        logger.info( "Number of cores to be used: %s, total available: %s\n", verified_jobs, max_jobs)
            
    return verified_jobs