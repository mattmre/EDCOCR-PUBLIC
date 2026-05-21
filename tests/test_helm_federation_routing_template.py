"""Tests for the Plan C Phase 1 / item C4 federation routing Helm overlay.

Verifies that:
* The default install (``federation.routing.enabled=false``) renders no extra
  routing env vars on the federation-overlay ConfigMap.
* When federation is enabled but routing is disabled, the
  ``OCR_FEDERATION_ROUTING_ENABLED`` flag renders as ``"false"``.
* When ``federation.routing.enabled=true`` plus a strategy override, the
  overlay ConfigMap exposes all five C4 env vars and ``coordinator``,
  ``gpu-worker``, ``cpu-worker`` deployments envFrom-mount the overlay
  (which is what surfaces the C4 env vars to those pods).

These tests shell out to the local ``helm`` CLI. They are skipped when
``helm`` is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = REPO_ROOT / "helm" / "ocr-local"
EXAMPLE_VALUES = CHART_PATH / "values-federation-example.yaml"

# Minimal secret overrides to bypass `required` template guards.
MIN_SECRETS = {
    "secrets.djangoSecretKey": "test-django-key",
    "secrets.postgresPassword": "test-pg",
    "secrets.rabbitmqPassword": "test-rmq",
    "secrets.redisPassword": "test-redis",
    "secrets.flowerPassword": "test-flower",
    "secrets.metricsApiKey": "test-metrics",
}


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _helm_template(*extra_args: str, values_files: list[Path] | None = None) -> str:
    cmd = ["helm", "template", "ocr-local", str(CHART_PATH)]
    for vf in values_files or []:
        cmd.extend(["--values", str(vf)])
    for k, v in MIN_SECRETS.items():
        cmd.extend(["--set", f"{k}={v}"])
    cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        pytest.fail(f"helm template failed: {proc.stderr}")
    return proc.stdout


def _parse_docs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if d]


def _federation_overlay_cm(docs: list[dict]) -> dict | None:
    for d in docs:
        if d.get("kind") != "ConfigMap":
            continue
        name = d.get("metadata", {}).get("name", "") or ""
        if name.endswith("-federation-overlay"):
            return d
    return None


def _coordinator_deployment(docs: list[dict]) -> dict | None:
    for d in docs:
        if d.get("kind") != "Deployment":
            continue
        if d.get("metadata", {}).get("name", "") == "ocr-local-coordinator":
            return d
    return None


def _deployment_by_name(docs: list[dict], name: str) -> dict | None:
    for d in docs:
        if d.get("kind") != "Deployment":
            continue
        if d.get("metadata", {}).get("name", "") == name:
            return d
    return None


pytestmark = pytest.mark.skipif(not _helm_available(), reason="helm CLI not on PATH")


# ---------------------------------------------------------------------------
# Default install (federation off)
# ---------------------------------------------------------------------------
def test_default_install_renders_no_overlay_configmap():
    """``federation.enabled=false`` (default) -> no overlay ConfigMap at all."""
    rendered = _helm_template()
    docs = _parse_docs(rendered)
    assert _federation_overlay_cm(docs) is None


def test_chart_default_routing_block_is_disabled():
    """``values.yaml`` must keep ``federation.routing.enabled`` false."""
    with open(CHART_PATH / "values.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    routing = data["federation"].get("routing", {})
    assert routing.get("enabled") is False
    assert routing.get("strategy") == "local"


# ---------------------------------------------------------------------------
# Federation enabled, routing disabled
# ---------------------------------------------------------------------------
def test_federation_enabled_routing_disabled_renders_disabled_flag():
    """With federation on but routing off, overlay carries ``ROUTING_ENABLED=false``."""
    rendered = _helm_template(values_files=[EXAMPLE_VALUES])
    docs = _parse_docs(rendered)
    overlay = _federation_overlay_cm(docs)
    assert overlay is not None
    data = overlay.get("data", {})
    # The example file does NOT enable routing, so flag must be false.
    assert data.get("OCR_FEDERATION_ROUTING_ENABLED") == "false"
    assert data.get("OCR_FEDERATION_ROUTING_STRATEGY") == "local"


# ---------------------------------------------------------------------------
# Federation + routing enabled
# ---------------------------------------------------------------------------
def test_routing_enabled_emits_all_env_vars():
    rendered = _helm_template(
        "--set", "federation.routing.enabled=true",
        "--set", "federation.routing.strategy=priority",
        "--set", "federation.routing.registry_poll_seconds=15",
        "--set", "federation.routing.load_aware_cache_seconds=7",
        "--set", "federation.routing.custom_callable=mypkg.router:pick",
        values_files=[EXAMPLE_VALUES],
    )
    docs = _parse_docs(rendered)
    overlay = _federation_overlay_cm(docs)
    assert overlay is not None
    data = overlay.get("data", {})
    assert data.get("OCR_FEDERATION_ROUTING_ENABLED") == "true"
    assert data.get("OCR_FEDERATION_ROUTING_STRATEGY") == "priority"
    assert data.get("OCR_FEDERATION_REGISTRY_POLL_SECONDS") == "15"
    assert data.get("OCR_FEDERATION_LOAD_AWARE_CACHE_SECONDS") == "7"
    assert data.get("OCR_FEDERATION_ROUTER_CALLABLE") == "mypkg.router:pick"


@pytest.mark.parametrize(
    "deployment_name",
    ["ocr-local-coordinator", "ocr-local-gpu-worker", "ocr-local-cpu-worker"],
)
def test_deployments_envfrom_overlay_when_routing_enabled(deployment_name):
    """All three pod kinds must envFrom-mount the overlay so the C4 env vars surface."""
    rendered = _helm_template(
        "--set", "federation.routing.enabled=true",
        "--set", "federation.routing.strategy=round_robin",
        values_files=[EXAMPLE_VALUES],
    )
    docs = _parse_docs(rendered)
    deployment = _deployment_by_name(docs, deployment_name)
    assert deployment is not None, f"missing deployment {deployment_name}"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env_from = container.get("envFrom", [])
    overlay_refs = [
        ef for ef in env_from
        if (ef.get("configMapRef") or {}).get("name", "").endswith(
            "-federation-overlay"
        )
    ]
    assert len(overlay_refs) == 1, (
        f"{deployment_name} must envFrom-mount federation overlay"
    )


def test_routing_disabled_renders_no_routing_envfrom_when_federation_off():
    """No federation -> no overlay -> no routing env vars on any deployment."""
    rendered = _helm_template()
    docs = _parse_docs(rendered)
    for name in ("ocr-local-coordinator", "ocr-local-gpu-worker", "ocr-local-cpu-worker"):
        d = _deployment_by_name(docs, name)
        assert d is not None
        env_from = d["spec"]["template"]["spec"]["containers"][0].get("envFrom", [])
        overlay_refs = [
            ef for ef in env_from
            if (ef.get("configMapRef") or {}).get("name", "").endswith(
                "-federation-overlay"
            )
        ]
        assert overlay_refs == [], (
            f"{name} must NOT envFrom-mount federation overlay when federation=off"
        )


# ---------------------------------------------------------------------------
# Strategy validation -- chart should accept all six declared strategies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "strategy",
    ["local", "priority", "round_robin", "load_aware", "region_affinity", "custom"],
)
def test_all_strategies_render(strategy):
    rendered = _helm_template(
        "--set", "federation.routing.enabled=true",
        "--set", f"federation.routing.strategy={strategy}",
        values_files=[EXAMPLE_VALUES],
    )
    docs = _parse_docs(rendered)
    overlay = _federation_overlay_cm(docs)
    assert overlay is not None
    assert overlay["data"]["OCR_FEDERATION_ROUTING_STRATEGY"] == strategy
