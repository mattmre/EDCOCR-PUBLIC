"""Federation control plane for the OCR_LOCAL multi-cluster swarm.

Plan C Phase 1, item C2 -- adds a small reconciler that watches a
ConfigMap-backed peer registry, probes each peer's RabbitMQ Management API,
and adds/removes federation Policy resources based on peer health.

The reconciler is deliberately lightweight: standard library HTTP, optional
``kubernetes`` client, and a Prometheus metrics server on port 9100.
"""

from __future__ import annotations

__all__ = ["reconciler", "cluster_router"]
