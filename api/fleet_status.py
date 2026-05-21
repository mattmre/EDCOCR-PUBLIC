"""Worker fleet status tracking with GPU utilization monitoring.

Tracks distributed worker fleet health including GPU utilization,
heartbeat monitoring, and job completion metrics for the operations
dashboard.  Thread-safe with global singleton access.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    ONLINE = "online"
    BUSY = "busy"
    IDLE = "idle"
    OFFLINE = "offline"
    DRAINING = "draining"
    ERROR = "error"


@dataclass
class GpuInfo:
    gpu_id: int = 0
    name: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    utilization_pct: float = 0.0
    temperature_c: float = 0.0

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)

    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return round((self.memory_used_mb / self.memory_total_mb) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "memory_utilization_pct": self.memory_utilization_pct,
            "utilization_pct": self.utilization_pct,
            "temperature_c": self.temperature_c,
        }


@dataclass
class WorkerInfo:
    worker_id: str = ""
    hostname: str = ""
    state: WorkerState = WorkerState.OFFLINE
    capabilities: list = field(default_factory=list)
    gpus: list = field(default_factory=list)  # List of GpuInfo
    current_job_id: str = ""
    jobs_completed: int = 0
    jobs_failed: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: float = 0.0
    queue_name: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.state in (WorkerState.ONLINE, WorkerState.BUSY, WorkerState.IDLE)

    @property
    def heartbeat_age_seconds(self) -> float:
        if self.last_heartbeat <= 0:
            return float("inf")
        return time.time() - self.last_heartbeat

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "state": self.state.value,
            "capabilities": self.capabilities,
            "gpus": [g.to_dict() for g in self.gpus],
            "current_job_id": self.current_job_id,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "last_heartbeat": self.last_heartbeat,
            "is_healthy": self.is_healthy,
            "queue_name": self.queue_name,
        }


@dataclass
class FleetSnapshot:
    timestamp: float = 0.0
    total_workers: int = 0
    online_workers: int = 0
    busy_workers: int = 0
    idle_workers: int = 0
    offline_workers: int = 0
    error_workers: int = 0
    draining_workers: int = 0
    total_gpus: int = 0
    avg_gpu_utilization_pct: float = 0.0
    avg_gpu_memory_pct: float = 0.0
    total_gpu_memory_mb: int = 0
    used_gpu_memory_mb: int = 0
    workers: list = field(default_factory=list)  # List of WorkerInfo

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_workers": self.total_workers,
                "online": self.online_workers,
                "busy": self.busy_workers,
                "idle": self.idle_workers,
                "offline": self.offline_workers,
                "error": self.error_workers,
                "draining": self.draining_workers,
            },
            "gpu": {
                "total_gpus": self.total_gpus,
                "avg_utilization_pct": round(self.avg_gpu_utilization_pct, 1),
                "avg_memory_pct": round(self.avg_gpu_memory_pct, 1),
                "total_memory_mb": self.total_gpu_memory_mb,
                "used_memory_mb": self.used_gpu_memory_mb,
            },
            "workers": [w.to_dict() for w in self.workers],
        }


class FleetTracker:
    """Thread-safe worker fleet status tracker."""

    def __init__(self, heartbeat_timeout: float = 60.0):
        self._lock = threading.Lock()
        self._workers: dict = {}  # worker_id -> WorkerInfo
        self._heartbeat_timeout = heartbeat_timeout

    def register_worker(
        self,
        worker_id: str,
        hostname: str = "",
        capabilities: list = None,
        queue_name: str = "",
        gpus: list = None,
    ):
        """Register a new worker or update existing registration."""
        with self._lock:
            info = self._workers.get(worker_id, WorkerInfo())
            info.worker_id = worker_id
            info.hostname = hostname or info.hostname
            info.capabilities = (
                capabilities if capabilities is not None else info.capabilities
            )
            info.queue_name = queue_name or info.queue_name
            info.state = WorkerState.ONLINE
            info.last_heartbeat = time.time()
            if gpus is not None:
                info.gpus = gpus
            self._workers[worker_id] = info

    def unregister_worker(self, worker_id: str):
        """Remove a worker from tracking."""
        with self._lock:
            self._workers.pop(worker_id, None)

    def heartbeat(
        self,
        worker_id: str,
        state: WorkerState = None,
        current_job_id: str = None,
        gpus: list = None,
    ):
        """Record a heartbeat from a worker."""
        with self._lock:
            if worker_id not in self._workers:
                return
            w = self._workers[worker_id]
            w.last_heartbeat = time.time()
            if state is not None:
                w.state = state
            if current_job_id is not None:
                w.current_job_id = current_job_id
            if gpus is not None:
                w.gpus = gpus

    def record_job_complete(self, worker_id: str, success: bool = True):
        """Record a completed job on a worker."""
        with self._lock:
            if worker_id not in self._workers:
                return
            w = self._workers[worker_id]
            if success:
                w.jobs_completed += 1
            else:
                w.jobs_failed += 1
            w.current_job_id = ""

    def update_gpu(
        self,
        worker_id: str,
        gpu_id: int,
        utilization_pct: float = None,
        memory_used_mb: int = None,
        temperature_c: float = None,
    ):
        """Update GPU metrics for a specific worker GPU."""
        with self._lock:
            if worker_id not in self._workers:
                return
            w = self._workers[worker_id]
            for g in w.gpus:
                if g.gpu_id == gpu_id:
                    if utilization_pct is not None:
                        g.utilization_pct = utilization_pct
                    if memory_used_mb is not None:
                        g.memory_used_mb = memory_used_mb
                    if temperature_c is not None:
                        g.temperature_c = temperature_c
                    break

    def check_stale_workers(self) -> list:
        """Mark workers with expired heartbeats as offline.

        Returns list of stale worker IDs.
        """
        now = time.time()
        stale = []
        with self._lock:
            for wid, w in self._workers.items():
                if w.last_heartbeat > 0 and (
                    now - w.last_heartbeat
                ) > self._heartbeat_timeout:
                    if w.state not in (WorkerState.OFFLINE, WorkerState.ERROR):
                        w.state = WorkerState.OFFLINE
                        stale.append(wid)
        return stale

    def get_worker(self, worker_id: str) -> WorkerInfo:
        """Get info for a specific worker."""
        with self._lock:
            w = self._workers.get(worker_id)
            if w is None:
                return None
            # Return a copy
            return WorkerInfo(
                worker_id=w.worker_id,
                hostname=w.hostname,
                state=w.state,
                capabilities=list(w.capabilities),
                gpus=list(w.gpus),
                current_job_id=w.current_job_id,
                jobs_completed=w.jobs_completed,
                jobs_failed=w.jobs_failed,
                uptime_seconds=w.uptime_seconds,
                last_heartbeat=w.last_heartbeat,
                queue_name=w.queue_name,
            )

    def get_snapshot(self) -> FleetSnapshot:
        """Get current fleet snapshot."""
        now = time.time()
        with self._lock:
            workers = list(self._workers.values())

        snap = FleetSnapshot(timestamp=now)
        snap.total_workers = len(workers)
        snap.workers = workers

        all_gpus = []
        for w in workers:
            if w.state == WorkerState.ONLINE:
                snap.online_workers += 1
            elif w.state == WorkerState.BUSY:
                snap.busy_workers += 1
            elif w.state == WorkerState.IDLE:
                snap.idle_workers += 1
            elif w.state == WorkerState.OFFLINE:
                snap.offline_workers += 1
            elif w.state == WorkerState.ERROR:
                snap.error_workers += 1
            elif w.state == WorkerState.DRAINING:
                snap.draining_workers += 1

            for g in w.gpus:
                all_gpus.append(g)

        snap.total_gpus = len(all_gpus)
        if all_gpus:
            snap.avg_gpu_utilization_pct = sum(
                g.utilization_pct for g in all_gpus
            ) / len(all_gpus)
            snap.avg_gpu_memory_pct = sum(
                g.memory_utilization_pct for g in all_gpus
            ) / len(all_gpus)
            snap.total_gpu_memory_mb = sum(g.memory_total_mb for g in all_gpus)
            snap.used_gpu_memory_mb = sum(g.memory_used_mb for g in all_gpus)

        return snap

    def reset(self):
        """Clear all tracked workers."""
        with self._lock:
            self._workers.clear()


# Global singleton
_tracker = None
_tracker_lock = threading.Lock()


def get_fleet_tracker() -> FleetTracker:
    """Return the global FleetTracker singleton, creating it if needed."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = FleetTracker()
    return _tracker
