import os

def get_threads_for_worker() -> int:
    """Return how many pyfftw threads this worker should use.

    When joblib spawns N worker processes each should use cpu_count/N threads
    so the total thread count stays close to the number of physical cores.
    joblib exposes the worker count via the LOKY_MAX_CPU_COUNT / joblib env
    variables; if we can't determine it we default to all cores.
    """
    cpu = os.cpu_count() or 1
    # joblib sets this env var in each worker process
    n_workers_str = os.environ.get("LOKY_MAX_CPU_COUNT") or os.environ.get("JOBLIB_NPROCS")
    try:
        n_workers = int(n_workers_str) if n_workers_str else 1
    except ValueError:
        n_workers = 1
    return max(1, cpu // n_workers)