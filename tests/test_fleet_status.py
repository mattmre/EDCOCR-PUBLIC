"""
Unit tests for worker fleet status tracking (api/fleet_status.py).

Tests cover:
- WorkerState enum values and count
- GpuInfo defaults and computed properties (memory_free_mb, memory_utilization_pct)
- GpuInfo.to_dict()
- GpuInfo memory_utilization_pct with zero total
- WorkerInfo defaults and properties (is_healthy, heartbeat_age_seconds)
- WorkerInfo.to_dict()
- FleetSnapshot defaults and to_dict()
- FleetTracker construction
- register_worker adds worker
- register_worker updates existing
- unregister_worker removes worker
- heartbeat updates last_heartbeat
- heartbeat updates state
- heartbeat for unknown worker (no-op)
- record_job_complete increments counts
- record_job_complete with failure
- record_job_complete unknown worker (no-op)
- update_gpu updates specific GPU
- update_gpu unknown worker (no-op)
- check_stale_workers marks stale as offline
- check_stale_workers returns stale IDs
- check_stale_workers ignores already offline
- get_worker returns copy
- get_worker unknown returns None
- get_snapshot empty fleet
- get_snapshot with workers and GPUs
- get_snapshot state counting (all 6 states)
- get_snapshot GPU aggregation
- reset clears all
- get_fleet_tracker singleton
- Thread safety concurrent operations

Run with: python -m pytest tests/test_fleet_status.py -v
"""

import threading
import time

# Add project root to path
from api.fleet_status import (
    FleetSnapshot,
    FleetTracker,
    GpuInfo,
    WorkerInfo,
    WorkerState,
    get_fleet_tracker,
)

# ---------------------------------------------------------------------------
# Tests: WorkerState
# ---------------------------------------------------------------------------


class TestWorkerState:
    def test_enum_values(self):
        assert WorkerState.ONLINE.value == "online"
        assert WorkerState.BUSY.value == "busy"
        assert WorkerState.IDLE.value == "idle"
        assert WorkerState.OFFLINE.value == "offline"
        assert WorkerState.DRAINING.value == "draining"
        assert WorkerState.ERROR.value == "error"

    def test_enum_count(self):
        assert len(WorkerState) == 6


# ---------------------------------------------------------------------------
# Tests: GpuInfo
# ---------------------------------------------------------------------------


class TestGpuInfo:
    def test_defaults(self):
        g = GpuInfo()
        assert g.gpu_id == 0
        assert g.name == ""
        assert g.memory_total_mb == 0
        assert g.memory_used_mb == 0
        assert g.utilization_pct == 0.0
        assert g.temperature_c == 0.0

    def test_memory_free_mb(self):
        g = GpuInfo(memory_total_mb=8192, memory_used_mb=3000)
        assert g.memory_free_mb == 5192

    def test_memory_free_mb_clamp(self):
        g = GpuInfo(memory_total_mb=1000, memory_used_mb=2000)
        assert g.memory_free_mb == 0

    def test_memory_utilization_pct(self):
        g = GpuInfo(memory_total_mb=10000, memory_used_mb=2500)
        assert g.memory_utilization_pct == 25.0

    def test_memory_utilization_pct_zero_total(self):
        g = GpuInfo(memory_total_mb=0, memory_used_mb=0)
        assert g.memory_utilization_pct == 0.0

    def test_to_dict(self):
        g = GpuInfo(
            gpu_id=1,
            name="RTX 4090",
            memory_total_mb=24576,
            memory_used_mb=8192,
            utilization_pct=75.5,
            temperature_c=68.0,
        )
        d = g.to_dict()
        assert d["gpu_id"] == 1
        assert d["name"] == "RTX 4090"
        assert d["memory_total_mb"] == 24576
        assert d["memory_used_mb"] == 8192
        assert d["memory_free_mb"] == 24576 - 8192
        assert d["utilization_pct"] == 75.5
        assert d["temperature_c"] == 68.0
        assert "memory_utilization_pct" in d

    def test_to_dict_keys(self):
        d = GpuInfo().to_dict()
        expected_keys = {
            "gpu_id",
            "name",
            "memory_total_mb",
            "memory_used_mb",
            "memory_free_mb",
            "memory_utilization_pct",
            "utilization_pct",
            "temperature_c",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: WorkerInfo
# ---------------------------------------------------------------------------


class TestWorkerInfo:
    def test_defaults(self):
        w = WorkerInfo()
        assert w.worker_id == ""
        assert w.hostname == ""
        assert w.state == WorkerState.OFFLINE
        assert w.capabilities == []
        assert w.gpus == []
        assert w.current_job_id == ""
        assert w.jobs_completed == 0
        assert w.jobs_failed == 0
        assert w.uptime_seconds == 0.0
        assert w.last_heartbeat == 0.0
        assert w.queue_name == ""

    def test_is_healthy_online(self):
        assert WorkerInfo(state=WorkerState.ONLINE).is_healthy is True

    def test_is_healthy_busy(self):
        assert WorkerInfo(state=WorkerState.BUSY).is_healthy is True

    def test_is_healthy_idle(self):
        assert WorkerInfo(state=WorkerState.IDLE).is_healthy is True

    def test_is_healthy_offline(self):
        assert WorkerInfo(state=WorkerState.OFFLINE).is_healthy is False

    def test_is_healthy_draining(self):
        assert WorkerInfo(state=WorkerState.DRAINING).is_healthy is False

    def test_is_healthy_error(self):
        assert WorkerInfo(state=WorkerState.ERROR).is_healthy is False

    def test_heartbeat_age_no_heartbeat(self):
        w = WorkerInfo(last_heartbeat=0)
        assert w.heartbeat_age_seconds == float("inf")

    def test_heartbeat_age_recent(self):
        w = WorkerInfo(last_heartbeat=time.time() - 5.0)
        age = w.heartbeat_age_seconds
        assert 4.0 <= age <= 7.0

    def test_to_dict(self):
        gpu = GpuInfo(gpu_id=0, name="A100")
        w = WorkerInfo(
            worker_id="w-1",
            hostname="node-a",
            state=WorkerState.BUSY,
            capabilities=["ocr", "pdf"],
            gpus=[gpu],
            current_job_id="job-42",
            jobs_completed=10,
            jobs_failed=2,
            uptime_seconds=3600.123,
            last_heartbeat=time.time(),
            queue_name="default",
        )
        d = w.to_dict()
        assert d["worker_id"] == "w-1"
        assert d["hostname"] == "node-a"
        assert d["state"] == "busy"
        assert d["capabilities"] == ["ocr", "pdf"]
        assert len(d["gpus"]) == 1
        assert d["current_job_id"] == "job-42"
        assert d["jobs_completed"] == 10
        assert d["jobs_failed"] == 2
        assert d["uptime_seconds"] == 3600.1
        assert d["is_healthy"] is True
        assert d["queue_name"] == "default"

    def test_to_dict_keys(self):
        d = WorkerInfo().to_dict()
        expected_keys = {
            "worker_id",
            "hostname",
            "state",
            "capabilities",
            "gpus",
            "current_job_id",
            "jobs_completed",
            "jobs_failed",
            "uptime_seconds",
            "last_heartbeat",
            "is_healthy",
            "queue_name",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: FleetSnapshot
# ---------------------------------------------------------------------------


class TestFleetSnapshot:
    def test_defaults(self):
        s = FleetSnapshot()
        assert s.timestamp == 0.0
        assert s.total_workers == 0
        assert s.online_workers == 0
        assert s.busy_workers == 0
        assert s.idle_workers == 0
        assert s.offline_workers == 0
        assert s.error_workers == 0
        assert s.draining_workers == 0
        assert s.total_gpus == 0
        assert s.avg_gpu_utilization_pct == 0.0
        assert s.avg_gpu_memory_pct == 0.0
        assert s.total_gpu_memory_mb == 0
        assert s.used_gpu_memory_mb == 0
        assert s.workers == []

    def test_to_dict_structure(self):
        s = FleetSnapshot(timestamp=1000.0, total_workers=2, online_workers=1)
        d = s.to_dict()
        assert d["timestamp"] == 1000.0
        assert "summary" in d
        assert "gpu" in d
        assert "workers" in d
        assert d["summary"]["total_workers"] == 2
        assert d["summary"]["online"] == 1

    def test_to_dict_gpu_section(self):
        s = FleetSnapshot(
            total_gpus=4,
            avg_gpu_utilization_pct=55.55,
            avg_gpu_memory_pct=33.33,
            total_gpu_memory_mb=49152,
            used_gpu_memory_mb=16384,
        )
        d = s.to_dict()
        gpu = d["gpu"]
        assert gpu["total_gpus"] == 4
        assert gpu["avg_utilization_pct"] == 55.5
        assert gpu["avg_memory_pct"] == 33.3
        assert gpu["total_memory_mb"] == 49152
        assert gpu["used_memory_mb"] == 16384


# ---------------------------------------------------------------------------
# Tests: FleetTracker
# ---------------------------------------------------------------------------


class TestFleetTracker:
    def test_construction(self):
        ft = FleetTracker(heartbeat_timeout=30.0)
        snap = ft.get_snapshot()
        assert snap.total_workers == 0

    def test_register_worker(self):
        ft = FleetTracker()
        ft.register_worker("w-1", hostname="node-a", capabilities=["ocr"])
        w = ft.get_worker("w-1")
        assert w is not None
        assert w.worker_id == "w-1"
        assert w.hostname == "node-a"
        assert w.capabilities == ["ocr"]
        assert w.state == WorkerState.ONLINE

    def test_register_worker_updates_existing(self):
        ft = FleetTracker()
        ft.register_worker("w-1", hostname="node-a")
        ft.register_worker("w-1", hostname="node-b", capabilities=["gpu"])
        w = ft.get_worker("w-1")
        assert w.hostname == "node-b"
        assert w.capabilities == ["gpu"]
        assert w.state == WorkerState.ONLINE

    def test_register_worker_with_gpus(self):
        ft = FleetTracker()
        gpus = [GpuInfo(gpu_id=0, name="A100", memory_total_mb=40960)]
        ft.register_worker("w-1", gpus=gpus)
        w = ft.get_worker("w-1")
        assert len(w.gpus) == 1
        assert w.gpus[0].name == "A100"

    def test_register_worker_preserves_fields(self):
        ft = FleetTracker()
        ft.register_worker("w-1", hostname="node-a", queue_name="high")
        # Re-register without hostname should keep existing
        ft.register_worker("w-1")
        w = ft.get_worker("w-1")
        assert w.hostname == "node-a"
        assert w.queue_name == "high"

    def test_unregister_worker(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        ft.unregister_worker("w-1")
        assert ft.get_worker("w-1") is None

    def test_unregister_unknown_worker(self):
        ft = FleetTracker()
        ft.unregister_worker("nonexistent")  # should not raise

    def test_heartbeat_updates_timestamp(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        time.sleep(0.05)
        before = time.time()
        ft.heartbeat("w-1")
        w = ft.get_worker("w-1")
        assert w.last_heartbeat >= before

    def test_heartbeat_updates_state(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        ft.heartbeat("w-1", state=WorkerState.BUSY, current_job_id="j-99")
        w = ft.get_worker("w-1")
        assert w.state == WorkerState.BUSY
        assert w.current_job_id == "j-99"

    def test_heartbeat_updates_gpus(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        new_gpus = [GpuInfo(gpu_id=0, name="V100")]
        ft.heartbeat("w-1", gpus=new_gpus)
        w = ft.get_worker("w-1")
        assert len(w.gpus) == 1
        assert w.gpus[0].name == "V100"

    def test_heartbeat_unknown_worker_noop(self):
        ft = FleetTracker()
        ft.heartbeat("unknown")  # should not raise

    def test_record_job_complete_success(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        ft.heartbeat("w-1", current_job_id="j-1")
        ft.record_job_complete("w-1", success=True)
        w = ft.get_worker("w-1")
        assert w.jobs_completed == 1
        assert w.jobs_failed == 0
        assert w.current_job_id == ""

    def test_record_job_complete_failure(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        ft.record_job_complete("w-1", success=False)
        w = ft.get_worker("w-1")
        assert w.jobs_completed == 0
        assert w.jobs_failed == 1

    def test_record_job_complete_multiple(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        ft.record_job_complete("w-1", success=True)
        ft.record_job_complete("w-1", success=True)
        ft.record_job_complete("w-1", success=False)
        w = ft.get_worker("w-1")
        assert w.jobs_completed == 2
        assert w.jobs_failed == 1

    def test_record_job_complete_unknown_worker_noop(self):
        ft = FleetTracker()
        ft.record_job_complete("unknown")  # should not raise

    def test_update_gpu(self):
        ft = FleetTracker()
        gpus = [
            GpuInfo(gpu_id=0, name="A100", memory_total_mb=40960),
            GpuInfo(gpu_id=1, name="A100", memory_total_mb=40960),
        ]
        ft.register_worker("w-1", gpus=gpus)
        ft.update_gpu("w-1", gpu_id=1, utilization_pct=85.0, memory_used_mb=20000, temperature_c=72.0)
        w = ft.get_worker("w-1")
        g1 = [g for g in w.gpus if g.gpu_id == 1][0]
        assert g1.utilization_pct == 85.0
        assert g1.memory_used_mb == 20000
        assert g1.temperature_c == 72.0
        # GPU 0 should be unchanged
        g0 = [g for g in w.gpus if g.gpu_id == 0][0]
        assert g0.utilization_pct == 0.0

    def test_update_gpu_partial(self):
        ft = FleetTracker()
        gpus = [GpuInfo(gpu_id=0, utilization_pct=50.0, temperature_c=60.0)]
        ft.register_worker("w-1", gpus=gpus)
        ft.update_gpu("w-1", gpu_id=0, utilization_pct=90.0)
        w = ft.get_worker("w-1")
        assert w.gpus[0].utilization_pct == 90.0
        assert w.gpus[0].temperature_c == 60.0  # unchanged

    def test_update_gpu_unknown_worker_noop(self):
        ft = FleetTracker()
        ft.update_gpu("unknown", gpu_id=0, utilization_pct=50.0)  # should not raise

    def test_update_gpu_unknown_gpu_id(self):
        ft = FleetTracker()
        gpus = [GpuInfo(gpu_id=0)]
        ft.register_worker("w-1", gpus=gpus)
        ft.update_gpu("w-1", gpu_id=99, utilization_pct=50.0)  # no match, no error
        w = ft.get_worker("w-1")
        assert w.gpus[0].utilization_pct == 0.0

    def test_check_stale_workers_marks_offline(self):
        ft = FleetTracker(heartbeat_timeout=0.05)
        ft.register_worker("w-1")
        time.sleep(0.1)
        stale = ft.check_stale_workers()
        assert "w-1" in stale
        w = ft.get_worker("w-1")
        assert w.state == WorkerState.OFFLINE

    def test_check_stale_workers_returns_ids(self):
        ft = FleetTracker(heartbeat_timeout=0.05)
        ft.register_worker("w-1")
        ft.register_worker("w-2")
        time.sleep(0.1)
        stale = ft.check_stale_workers()
        assert set(stale) == {"w-1", "w-2"}

    def test_check_stale_workers_ignores_already_offline(self):
        ft = FleetTracker(heartbeat_timeout=0.05)
        ft.register_worker("w-1")
        time.sleep(0.1)
        ft.check_stale_workers()  # marks offline
        stale2 = ft.check_stale_workers()  # already offline
        assert stale2 == []

    def test_check_stale_workers_ignores_error_state(self):
        ft = FleetTracker(heartbeat_timeout=0.05)
        ft.register_worker("w-1")
        ft.heartbeat("w-1", state=WorkerState.ERROR)
        time.sleep(0.1)
        stale = ft.check_stale_workers()
        assert stale == []

    def test_check_stale_workers_fresh_not_stale(self):
        ft = FleetTracker(heartbeat_timeout=60.0)
        ft.register_worker("w-1")
        stale = ft.check_stale_workers()
        assert stale == []

    def test_get_worker_returns_copy(self):
        ft = FleetTracker()
        ft.register_worker("w-1", capabilities=["ocr"])
        w = ft.get_worker("w-1")
        w.capabilities.append("pdf")
        w2 = ft.get_worker("w-1")
        assert "pdf" not in w2.capabilities

    def test_get_worker_unknown_returns_none(self):
        ft = FleetTracker()
        assert ft.get_worker("nonexistent") is None

    def test_get_snapshot_empty_fleet(self):
        ft = FleetTracker()
        snap = ft.get_snapshot()
        assert snap.total_workers == 0
        assert snap.total_gpus == 0
        assert snap.avg_gpu_utilization_pct == 0.0
        assert snap.workers == []
        assert snap.timestamp > 0

    def test_get_snapshot_with_workers_and_gpus(self):
        ft = FleetTracker()
        gpus = [GpuInfo(gpu_id=0, memory_total_mb=8192, memory_used_mb=4096, utilization_pct=50.0)]
        ft.register_worker("w-1", gpus=gpus)
        ft.register_worker("w-2")
        snap = ft.get_snapshot()
        assert snap.total_workers == 2
        assert snap.total_gpus == 1
        assert snap.online_workers == 2

    def test_get_snapshot_state_counting(self):
        ft = FleetTracker()
        ft.register_worker("w-online")
        ft.register_worker("w-busy")
        ft.heartbeat("w-busy", state=WorkerState.BUSY)
        ft.register_worker("w-idle")
        ft.heartbeat("w-idle", state=WorkerState.IDLE)
        ft.register_worker("w-offline")
        ft.heartbeat("w-offline", state=WorkerState.OFFLINE)
        ft.register_worker("w-error")
        ft.heartbeat("w-error", state=WorkerState.ERROR)
        ft.register_worker("w-drain")
        ft.heartbeat("w-drain", state=WorkerState.DRAINING)

        snap = ft.get_snapshot()
        assert snap.total_workers == 6
        assert snap.online_workers == 1
        assert snap.busy_workers == 1
        assert snap.idle_workers == 1
        assert snap.offline_workers == 1
        assert snap.error_workers == 1
        assert snap.draining_workers == 1

    def test_get_snapshot_gpu_aggregation(self):
        ft = FleetTracker()
        gpus_a = [
            GpuInfo(gpu_id=0, memory_total_mb=10000, memory_used_mb=2000, utilization_pct=40.0),
            GpuInfo(gpu_id=1, memory_total_mb=10000, memory_used_mb=6000, utilization_pct=80.0),
        ]
        gpus_b = [
            GpuInfo(gpu_id=0, memory_total_mb=20000, memory_used_mb=10000, utilization_pct=60.0),
        ]
        ft.register_worker("w-1", gpus=gpus_a)
        ft.register_worker("w-2", gpus=gpus_b)
        snap = ft.get_snapshot()
        assert snap.total_gpus == 3
        assert snap.total_gpu_memory_mb == 40000
        assert snap.used_gpu_memory_mb == 18000
        # avg utilization: (40 + 80 + 60) / 3 = 60.0
        assert abs(snap.avg_gpu_utilization_pct - 60.0) < 0.1

    def test_get_snapshot_to_dict(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        snap = ft.get_snapshot()
        d = snap.to_dict()
        assert "timestamp" in d
        assert "summary" in d
        assert "gpu" in d
        assert "workers" in d
        assert isinstance(d["workers"], list)
        assert len(d["workers"]) == 1

    def test_reset(self):
        ft = FleetTracker()
        ft.register_worker("w-1")
        ft.register_worker("w-2")
        ft.reset()
        snap = ft.get_snapshot()
        assert snap.total_workers == 0

    def test_get_fleet_tracker_singleton(self):
        t1 = get_fleet_tracker()
        t2 = get_fleet_tracker()
        assert t1 is t2
        assert isinstance(t1, FleetTracker)

    def test_thread_safety_concurrent_operations(self):
        ft = FleetTracker()
        errors = []
        barrier = threading.Barrier(4)

        def register_workers(start_id, count):
            try:
                barrier.wait(timeout=5)
                for i in range(count):
                    wid = f"w-{start_id + i}"
                    gpus = [GpuInfo(gpu_id=0, memory_total_mb=8192)]
                    ft.register_worker(wid, hostname=f"node-{wid}", gpus=gpus)
                    ft.heartbeat(wid, state=WorkerState.BUSY)
                    ft.record_job_complete(wid, success=True)
                    ft.update_gpu(wid, gpu_id=0, utilization_pct=50.0)
                    ft.get_snapshot()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_workers, args=(0, 25)),
            threading.Thread(target=register_workers, args=(100, 25)),
            threading.Thread(target=register_workers, args=(200, 25)),
            threading.Thread(target=register_workers, args=(300, 25)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        snap = ft.get_snapshot()
        assert snap.total_workers == 100
