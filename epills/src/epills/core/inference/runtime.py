"""The CPUs this process can really use (latency: too many threads on a container's few cores slow PyTorch down)."""
import os


def cpu_cores():
    """The container's CPU quota (cgroup v2, then v1), else the CPUs this process may run on, else os.cpu_count()."""
    for quota_file, period_file in (("/sys/fs/cgroup/cpu.max", None),
                                    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            if period_file is None:
                quota, period = open(quota_file).read().split()[:2]
            else:
                quota, period = open(quota_file).read().strip(), open(period_file).read().strip()
            if quota not in ("max", "-1"):
                return max(1, int(int(quota) / int(period)))
        except (OSError, ValueError):
            pass
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)
