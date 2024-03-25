import multiprocessing
import os


def create_pool() -> multiprocessing.Pool:
    """Create a multiprocessing pool configured with the number of available CPU cores.
    Will automatically detect if the code is being run through Slurm on an HPC cluster node.
    """
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        num_processes = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        print('No environment setting, using all CPU cores.')
        num_processes = multiprocessing.cpu_count()
    print(f'Running with {num_processes} processes.')
    pool = multiprocessing.Pool(num_processes)
    return pool
