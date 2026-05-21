"""Job queue depth monitoring and alerting.

Tracks queue depths across pipeline stages, detects threshold
breaches, and fires configurable alerts via callback functions.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class QueueThreshold:
    """Threshold configuration for a queue."""
    queue_name: str
    warning_depth: int = 50
    critical_depth: int = 100
    warning_wait_seconds: float = 300.0
    critical_wait_seconds: float = 600.0

    def to_dict(self) -> dict:
        return {
            "queue_name": self.queue_name,
            "warning_depth": self.warning_depth,
            "critical_depth": self.critical_depth,
            "warning_wait_seconds": self.warning_wait_seconds,
            "critical_wait_seconds": self.critical_wait_seconds,
        }


@dataclass
class QueueDepthPoint:
    """A point-in-time queue depth measurement."""
    timestamp: float
    queue_name: str
    depth: int
    oldest_item_age_seconds: float = 0.0


@dataclass
class Alert:
    """An alert triggered by threshold breach."""
    alert_id: str
    queue_name: str
    severity: AlertSeverity
    state: AlertState = AlertState.ACTIVE
    message: str = ""
    triggered_at: float = 0.0
    resolved_at: float = 0.0
    acknowledged_at: float = 0.0
    current_depth: int = 0
    threshold_value: int = 0

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "queue_name": self.queue_name,
            "severity": self.severity.value,
            "state": self.state.value,
            "message": self.message,
            "triggered_at": self.triggered_at,
            "resolved_at": self.resolved_at,
            "acknowledged_at": self.acknowledged_at,
            "current_depth": self.current_depth,
            "threshold_value": self.threshold_value,
        }


@dataclass
class QueueSnapshot:
    """Current state of all monitored queues."""
    timestamp: float = 0.0
    queues: list = field(default_factory=list)  # List of dicts with depth info
    active_alerts: list = field(default_factory=list)  # List of Alert
    total_depth: int = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_depth": self.total_depth,
            "queues": self.queues,
            "active_alerts": [a.to_dict() for a in self.active_alerts],
        }


class QueueMonitor:
    """Thread-safe queue depth monitor with alerting."""

    def __init__(self, history_seconds: int = 3600, thresholds_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._history_seconds = history_seconds
        self._thresholds_path = Path(thresholds_path) if thresholds_path else None
        self._thresholds: dict = {}  # queue_name -> QueueThreshold
        self._history: dict = {}  # queue_name -> deque of QueueDepthPoint
        self._current_depths: dict = {}  # queue_name -> int
        self._alerts: dict = {}  # alert_id -> Alert
        self._alert_counter = 0
        self._alert_callbacks: list = []  # List of callable(Alert)
        self._load_thresholds()

    def set_threshold(self, queue_name: str, warning_depth: int = 50,
                      critical_depth: int = 100,
                      warning_wait_seconds: float = 300.0,
                      critical_wait_seconds: float = 600.0):
        """Set alerting thresholds for a queue."""
        with self._lock:
            self._thresholds[queue_name] = QueueThreshold(
                queue_name=queue_name,
                warning_depth=warning_depth,
                critical_depth=critical_depth,
                warning_wait_seconds=warning_wait_seconds,
                critical_wait_seconds=critical_wait_seconds,
            )
            self._persist_thresholds_locked()

    def get_threshold(self, queue_name: str) -> QueueThreshold | None:
        """Return the configured alerting threshold for a queue, if present."""
        with self._lock:
            threshold = self._thresholds.get(queue_name)
            if threshold is None:
                return None
            return QueueThreshold(
                queue_name=threshold.queue_name,
                warning_depth=threshold.warning_depth,
                critical_depth=threshold.critical_depth,
                warning_wait_seconds=threshold.warning_wait_seconds,
                critical_wait_seconds=threshold.critical_wait_seconds,
            )

    def list_thresholds(self) -> list[QueueThreshold]:
        """Return configured queue alert thresholds."""
        with self._lock:
            return [
                QueueThreshold(
                    queue_name=threshold.queue_name,
                    warning_depth=threshold.warning_depth,
                    critical_depth=threshold.critical_depth,
                    warning_wait_seconds=threshold.warning_wait_seconds,
                    critical_wait_seconds=threshold.critical_wait_seconds,
                )
                for threshold in self._thresholds.values()
            ]

    def add_alert_callback(self, callback):
        """Register a callback function for alerts. Called with Alert object."""
        with self._lock:
            self._alert_callbacks.append(callback)

    def record_depth(self, queue_name: str, depth: int,
                     oldest_item_age_seconds: float = 0.0):
        """Record current queue depth. Checks thresholds and fires alerts."""
        now = time.time()
        point = QueueDepthPoint(
            timestamp=now,
            queue_name=queue_name,
            depth=depth,
            oldest_item_age_seconds=oldest_item_age_seconds,
        )

        callbacks_to_fire = []

        with self._lock:
            # Store in history
            if queue_name not in self._history:
                self._history[queue_name] = deque()
            self._history[queue_name].append(point)
            self._current_depths[queue_name] = depth

            # Prune old
            cutoff = now - self._history_seconds
            while self._history[queue_name] and self._history[queue_name][0].timestamp < cutoff:
                self._history[queue_name].popleft()

            # Check thresholds
            threshold = self._thresholds.get(queue_name)
            if threshold:
                # Check for critical
                if depth >= threshold.critical_depth:
                    alert = self._create_alert(
                        queue_name, AlertSeverity.CRITICAL,
                        f"Queue '{queue_name}' depth {depth} exceeds critical threshold {threshold.critical_depth}",
                        depth, threshold.critical_depth,
                    )
                    if alert:
                        callbacks_to_fire.append(alert)
                elif depth >= threshold.warning_depth:
                    alert = self._create_alert(
                        queue_name, AlertSeverity.WARNING,
                        f"Queue '{queue_name}' depth {depth} exceeds warning threshold {threshold.warning_depth}",
                        depth, threshold.warning_depth,
                    )
                    if alert:
                        callbacks_to_fire.append(alert)
                else:
                    # Below threshold — auto-resolve active alerts for this queue
                    self._auto_resolve_alerts(queue_name)

                # Check wait time thresholds
                if oldest_item_age_seconds >= threshold.critical_wait_seconds:
                    alert = self._create_alert(
                        queue_name, AlertSeverity.CRITICAL,
                        f"Queue '{queue_name}' oldest item {oldest_item_age_seconds:.0f}s exceeds critical wait {threshold.critical_wait_seconds:.0f}s",
                        depth, 0,
                    )
                    if alert:
                        callbacks_to_fire.append(alert)
                elif oldest_item_age_seconds >= threshold.warning_wait_seconds:
                    alert = self._create_alert(
                        queue_name, AlertSeverity.WARNING,
                        f"Queue '{queue_name}' oldest item {oldest_item_age_seconds:.0f}s exceeds warning wait {threshold.warning_wait_seconds:.0f}s",
                        depth, 0,
                    )
                    if alert:
                        callbacks_to_fire.append(alert)

            callbacks = list(self._alert_callbacks)

        # Fire callbacks outside lock
        for alert in callbacks_to_fire:
            for cb in callbacks:
                try:
                    cb(alert)
                except Exception:
                    logger.exception("Alert callback failed")

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an active alert."""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.state == AlertState.ACTIVE:
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_at = time.time()

    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert."""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.state != AlertState.RESOLVED:
                alert.state = AlertState.RESOLVED
                alert.resolved_at = time.time()

    def get_active_alerts(self) -> list:
        """Get all non-resolved alerts."""
        with self._lock:
            return [a for a in self._alerts.values() if a.state != AlertState.RESOLVED]

    def get_queue_history(self, queue_name: str, window_seconds: int = 3600) -> list:
        """Get depth history for a queue as time-series data."""
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            points = self._history.get(queue_name, [])
            return [
                {
                    "timestamp": p.timestamp,
                    "depth": p.depth,
                    "oldest_item_age_seconds": p.oldest_item_age_seconds,
                }
                for p in points
                if p.timestamp >= cutoff
            ]

    def get_snapshot(self) -> QueueSnapshot:
        """Get current state of all monitored queues."""
        now = time.time()
        with self._lock:
            queues = []
            total = 0
            for qname, depth in self._current_depths.items():
                threshold = self._thresholds.get(qname)
                queues.append({
                    "queue_name": qname,
                    "depth": depth,
                    "warning_threshold": threshold.warning_depth if threshold else None,
                    "critical_threshold": threshold.critical_depth if threshold else None,
                    "warning_wait_seconds": threshold.warning_wait_seconds if threshold else None,
                    "critical_wait_seconds": threshold.critical_wait_seconds if threshold else None,
                })
                total += depth

            active = [a for a in self._alerts.values() if a.state != AlertState.RESOLVED]

            return QueueSnapshot(
                timestamp=now,
                queues=queues,
                active_alerts=active,
                total_depth=total,
            )

    def reset(self):
        """Clear all data."""
        with self._lock:
            self._thresholds.clear()
            self._history.clear()
            self._current_depths.clear()
            self._alerts.clear()
            self._alert_counter = 0
            self._alert_callbacks.clear()
            self._persist_thresholds_locked()

    def _load_thresholds(self) -> None:
        if self._thresholds_path is None or not self._thresholds_path.exists():
            return
        try:
            raw = json.loads(self._thresholds_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load queue threshold config from %s", self._thresholds_path)
            return
        if not isinstance(raw, list):
            logger.warning("Queue threshold config ignored: expected list in %s", self._thresholds_path)
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                threshold = QueueThreshold(
                    queue_name=str(item["queue_name"]),
                    warning_depth=int(item["warning_depth"]),
                    critical_depth=int(item["critical_depth"]),
                    warning_wait_seconds=float(item["warning_wait_seconds"]),
                    critical_wait_seconds=float(item["critical_wait_seconds"]),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("Queue threshold config entry ignored: %r", item)
                continue
            self._thresholds[threshold.queue_name] = threshold

    def _persist_thresholds_locked(self) -> None:
        if self._thresholds_path is None:
            return
        payload = [
            threshold.to_dict()
            for threshold in sorted(
                self._thresholds.values(),
                key=lambda item: item.queue_name,
            )
        ]
        try:
            self._thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            self._thresholds_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to persist queue threshold config to %s", self._thresholds_path)

    def _create_alert(self, queue_name: str, severity: AlertSeverity,
                      message: str, depth: int, threshold_value: int) -> Alert:
        """Create alert if no active alert exists for this queue+severity. Returns Alert or None."""
        # Check if active alert already exists for queue+severity
        for a in self._alerts.values():
            if (a.queue_name == queue_name and a.severity == severity
                    and a.state != AlertState.RESOLVED):
                # Update existing
                a.current_depth = depth
                return None

        self._alert_counter += 1
        alert_id = f"alert-{self._alert_counter}"
        alert = Alert(
            alert_id=alert_id,
            queue_name=queue_name,
            severity=severity,
            message=message,
            triggered_at=time.time(),
            current_depth=depth,
            threshold_value=threshold_value,
        )
        self._alerts[alert_id] = alert
        return alert

    def _auto_resolve_alerts(self, queue_name: str):
        """Auto-resolve active alerts for a queue that's back below threshold."""
        now = time.time()
        for a in self._alerts.values():
            if a.queue_name == queue_name and a.state != AlertState.RESOLVED:
                a.state = AlertState.RESOLVED
                a.resolved_at = now


# Global singleton
_monitor = None
_monitor_lock = threading.Lock()


def get_queue_monitor() -> QueueMonitor:
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = QueueMonitor(thresholds_path=_default_thresholds_path())
    return _monitor


def _default_thresholds_path() -> Path:
    configured = os.environ.get("OCR_QUEUE_THRESHOLDS_PATH", "").strip()
    if configured:
        return Path(configured)
    try:
        from api.config import OUTPUT_FOLDER

        return Path(OUTPUT_FOLDER) / "queue_thresholds.json"
    except Exception:
        return Path("out") / "queue_thresholds.json"
