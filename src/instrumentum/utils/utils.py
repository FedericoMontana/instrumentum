import logging
import multiprocessing
from itertools import chain, combinations

logger = logging.getLogger(__name__)


def get_combs(set_size, combs_to, combs_from=1):

    if combs_from > combs_to or combs_from < 0:
        raise ValueError("combs_to must be possitive and less or equal to combs_from")
    # We accepts combs_to > set_size, it will just use set_size as the limit

    l_comb = [combinations(range(set_size), x) for x in range(combs_from, combs_to + 1)]

    return tuple(chain.from_iterable(l_comb))


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
        logger.error(
            "Jobs cannot be negative (except for -1 for all cores). Your total cores \
            are %s , modifying your entry to 1",
            max_jobs,
        )
        verified_jobs = 1

    else:
        if n_jobs == -1:
            verified_jobs = max_jobs

        logger.info(
            "Number of cores to be used: %s, total available: %s\n",
            verified_jobs,
            max_jobs,
        )

    return verified_jobs
