"""Unit tests for the federation cluster registry (Plan C Phase 1, item C4).

Exercises the YAML-file mode used by dev/test environments and the polling
+ live-reload semantics of ``ClusterRegistry``. ConfigMap mode is exercised
via a mock ``read_namespaced_config_map`` so the test suite stays
dependency-free.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from federation.cluster_router import ClusterRegistry

YAML_DOC = """\
self:
  name: prod-us-east-1
clusters:
  - name: prod-eu-central-1
    region: eu-central-1
    rabbitmq_uri: amqps://user@rabbit.eu/%2F
    management_uri: https://rabbit.eu:15671
    priority: 10
    tags: [commercial]
    tls:
      enabled: true
      ca_secret_ref: rabbitmq-fed-eu-ca
  - name: airgap-gov-1
    region: on-prem-gov
    rabbitmq_uri: amqps://user@rabbit.gov/%2F
    management_uri: https://rabbit.gov:15671
    priority: 5
    tags: [privileged, airgap]
    tls:
      enabled: true
      ca_secret_ref: rabbitmq-fed-gov-ca
"""


JSON_DOC = """\
{
  "self": {"name": "prod-us-east-1"},
  "clusters": [
    {
      "name": "prod-eu-central-1",
      "region": "eu-central-1",
      "rabbitmq_uri": "amqps://user@rabbit.eu/%2F",
      "management_uri": "https://rabbit.eu:15671",
      "priority": 10,
      "tags": ["commercial"]
    }
  ]
}
"""


def test_yaml_file_mode_parses_full_document(tmp_path):
    path = tmp_path / "clusters.yaml"
    path.write_text(YAML_DOC, encoding="utf-8")
    reg = ClusterRegistry(path=str(path))
    clusters = reg.clusters()
    assert len(clusters) == 2
    assert reg.self_name() == "prod-us-east-1"
    eu = next(c for c in clusters if c.name == "prod-eu-central-1")
    assert eu.region == "eu-central-1"
    assert eu.priority == 10
    assert eu.tags == ("commercial",)
    assert eu.tls_enabled is True


def test_json_file_mode_also_works(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(JSON_DOC, encoding="utf-8")
    reg = ClusterRegistry(path=str(path))
    clusters = reg.clusters()
    assert len(clusters) == 1
    assert clusters[0].name == "prod-eu-central-1"


def test_missing_file_yields_empty_registry(tmp_path):
    path = tmp_path / "nonexistent.yaml"
    reg = ClusterRegistry(path=str(path))
    assert reg.clusters() == []
    assert reg.self_name() == ""


def test_polling_picks_up_live_edits(tmp_path):
    path = tmp_path / "clusters.yaml"
    path.write_text(
        "self:\n  name: lt\nclusters:\n  - name: a\n    region: r1\n    priority: 1\n",
        encoding="utf-8",
    )
    fake_now = [1000.0]
    reg = ClusterRegistry(
        path=str(path),
        poll_seconds=10.0,
        clock=lambda: fake_now[0],
    )
    assert {c.name for c in reg.clusters()} == {"a"}

    # Edit the file but stay within the poll window: registry should NOT
    # reload yet.
    path.write_text(
        "self:\n  name: lt\nclusters:\n  - name: a\n    region: r1\n    priority: 1\n  - name: b\n    region: r2\n    priority: 2\n",
        encoding="utf-8",
    )
    fake_now[0] += 1.0
    assert {c.name for c in reg.clusters()} == {"a"}

    # Advance past the poll window: registry reloads.
    fake_now[0] += 30.0
    assert {c.name for c in reg.clusters()} == {"a", "b"}


def test_force_reload_bypasses_poll_window(tmp_path):
    path = tmp_path / "clusters.yaml"
    path.write_text(
        "self:\n  name: lt\nclusters:\n  - name: a\n    region: r1\n    priority: 1\n",
        encoding="utf-8",
    )
    fake_now = [1000.0]
    reg = ClusterRegistry(
        path=str(path),
        poll_seconds=120.0,
        clock=lambda: fake_now[0],
    )
    assert {c.name for c in reg.clusters()} == {"a"}

    path.write_text(
        "self:\n  name: lt\nclusters:\n  - name: c\n    region: r3\n    priority: 9\n",
        encoding="utf-8",
    )
    # Without force_reload the cached value remains (poll_seconds=120).
    assert {c.name for c in reg.clusters()} == {"a"}
    reg.force_reload()
    assert {c.name for c in reg.clusters()} == {"c"}


def test_health_provider_is_consulted():
    reg = ClusterRegistry(
        path=None,
        health_provider=lambda name: name == "ok",
    )
    assert reg.is_peer_healthy("ok") is True
    assert reg.is_peer_healthy("other") is False


def test_default_health_provider_returns_true(tmp_path):
    path = tmp_path / "clusters.yaml"
    path.write_text("self:\n  name: lt\nclusters: []\n", encoding="utf-8")
    reg = ClusterRegistry(path=str(path))
    # Default provider is "always healthy" so peer-router strategies don't
    # silently drop everything when no Prometheus scrape is wired in.
    assert reg.is_peer_healthy("anything") is True


def test_configmap_mode_reads_from_kubernetes_api():
    # Mock a CoreV1Api whose read_namespaced_config_map returns a simple
    # data dict containing clusters.json.
    api = MagicMock()
    cm = SimpleNamespace(
        data={
            "clusters.json": JSON_DOC,
        }
    )
    api.read_namespaced_config_map.return_value = cm
    reg = ClusterRegistry(
        configmap_name="federation-cluster-registry",
        configmap_namespace="default",
        kubernetes_api=api,
    )
    clusters = reg.clusters()
    assert len(clusters) == 1
    assert clusters[0].name == "prod-eu-central-1"
    api.read_namespaced_config_map.assert_called_once()


def test_configmap_mode_handles_missing_data():
    api = MagicMock()
    cm = SimpleNamespace(data={})
    api.read_namespaced_config_map.return_value = cm
    reg = ClusterRegistry(
        configmap_name="federation-cluster-registry",
        configmap_namespace="default",
        kubernetes_api=api,
    )
    assert reg.clusters() == []


def test_configmap_mode_handles_api_failure():
    api = MagicMock()
    api.read_namespaced_config_map.side_effect = RuntimeError("boom")
    reg = ClusterRegistry(
        configmap_name="federation-cluster-registry",
        configmap_namespace="default",
        kubernetes_api=api,
    )
    # Failed reads keep the registry empty rather than blowing up.
    assert reg.clusters() == []


def test_unparseable_file_keeps_registry_empty(tmp_path):
    path = tmp_path / "clusters.yaml"
    path.write_text("[this is :: not :: valid YAML or JSON\n", encoding="utf-8")
    reg = ClusterRegistry(path=str(path))
    assert reg.clusters() == []
