"""Terminate only a job's owned subprocess tree."""

import psutil


def terminate_tree(pid: int) -> None:
    """Terminate descendants and their parent, escalating after a grace period.

    Example:
        >>> terminate_tree(job_subprocess.pid)
    """
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=3)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        return
