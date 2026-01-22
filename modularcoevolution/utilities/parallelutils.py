#  Copyright 2026 BONSAI Lab at Auburn University
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

import concurrent.futures
import itertools
import logging
import multiprocessing
import os
import sys
import warnings


_logger = logging.getLogger(__name__)


def cores_available() -> int:
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        num_processes = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    elif hasattr(os, 'process_cpu_count'):  # Added in Python 3.13, should be equivalent to the above.
        num_processes = os.process_cpu_count()
    else:
        warnings.warn('No environment setting, using all CPU cores.')
        num_processes = multiprocessing.cpu_count()
    return num_processes


def create_pool(num_processes: int = -1) -> concurrent.futures.Executor:
    """Create an Executor pool configured with the number of available CPU cores.
    If the GIL is disabled, uses a ThreadPoolExecutor instead of a ProcessPoolExecutor.
    Will automatically detect if the code is being run through Slurm on an HPC cluster node
    and use the assigned number of cores.
    """
    # `sys._is_gil_enabled` was added in Python 3.13, but we want to use multiprocessing in earlier versions anyway.
    use_multiprocessing = not hasattr(sys, '_is_gil_enabled') or sys._is_gil_enabled()

    if num_processes == -1:
        num_processes = cores_available()

    if use_multiprocessing:
        _logger.info(f'Creating pool with {num_processes} processes.')
        # ProcessPoolExecutor handles failure much worse than multiprocessing.Pool.
        # return concurrent.futures.ProcessPoolExecutor(max_workers=num_processes)
        return MultiprocessingPoolWrapper(processes=num_processes)
    else:
        _logger.info(f'Creating pool with {num_processes} threads (GIL is disabled).')
        return concurrent.futures.ThreadPoolExecutor(max_workers=num_processes)


class MultiprocessingPoolWrapper:
    """A simple wrapper around a multiprocessing.Pool to provide a similar interface to
    concurrent.futures.Executor.
    """
    pool: multiprocessing.Pool

    def __init__(self, processes: int):
        self.pool = multiprocessing.Pool(processes=processes)

    def map(self, fn, *iterables, chunksize: int = 1):
        # concurrent.futures.Executor groups by argument position, then task index.
        # Meanwhile, multiprocessing.Pool.starmap groups by task index, then argument position.
        starmap_iterables = zip(*iterables)
        wrapper_iterables = ((fn, args) for args in starmap_iterables)
        return self.pool.imap(_starmap_wrapper, wrapper_iterables, chunksize=chunksize)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        if cancel_futures:
            self.pool.terminate()
        else:
            self.pool.close()
        if wait:
            self.pool.join()

    def __enter__(self):
        return self.pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.pool.close()
        else:
            self.pool.terminate()
        self.pool.join()
        return False


def _starmap_wrapper(fn_and_args):
    """For use with `multiprocessing.Pool.imap` to make it function like `starmap`."""
    function, args = fn_and_args
    return function(*args)