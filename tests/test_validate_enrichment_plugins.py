"""Tests for the enrichment plugin entry-point validator."""

from __future__ import annotations

from dataclasses import replace

from ocr_local.enrichment_bus import ResourceLimits, UIDescriptor, sign_descriptor
from scripts import validate_enrichment_plugins as validator


class Provider:
    def __init__(self, descriptor: UIDescriptor) -> None:
        self._descriptor = descriptor

    def ui_descriptor(self) -> UIDescriptor:
        return self._descriptor


def _descriptor(signature=True) -> UIDescriptor:
    descriptor = UIDescriptor(
        plugin_id="sample",
        display_name="Sample",
        protocol_version="1.0",
        signature=None,
        resource_limits=ResourceLimits(64, 50, 1000),
        deterministic=True,
        network_required=False,
        trust_tier="sandboxed",
        publisher="core",
        publisher_type="core",
    )
    if signature:
        descriptor = replace(descriptor, signature=sign_descriptor(descriptor, "secret"))
    return descriptor


def test_validator_passes_when_discovered_plugin_admits(monkeypatch):
    monkeypatch.setenv("CORE_PLUGIN_KEY", "secret")
    monkeypatch.setattr(
        validator,
        "discover_entry_point_providers",
        lambda group: [Provider(_descriptor())],
    )

    assert validator.main(["--publisher", "core=CORE_PLUGIN_KEY"]) == 0


def test_validator_require_plugin_passes_when_discovered_plugin_admits(monkeypatch):
    monkeypatch.setenv("CORE_PLUGIN_KEY", "secret")
    monkeypatch.setattr(
        validator,
        "discover_entry_point_providers",
        lambda group: [Provider(_descriptor())],
    )

    assert (
        validator.main(["--publisher", "core=CORE_PLUGIN_KEY", "--require-plugin"])
        == 0
    )


def test_validator_require_plugin_passes_with_fixture_provider(monkeypatch):
    monkeypatch.setenv("FIXTURE_PLUGIN_KEY", "fixture-secret")
    monkeypatch.setattr(
        validator,
        "discover_entry_point_providers",
        lambda group: [],
    )

    assert (
        validator.main(
            [
                "--publisher",
                "fixture=FIXTURE_PLUGIN_KEY",
                "--provider",
                "tests.fixtures.enrichment_plugins.signed_provider:build_provider",
                "--require-plugin",
            ]
        )
        == 0
    )


def test_validator_empty_discovery_with_require_plugin_fails(monkeypatch):
    monkeypatch.setattr(
        validator,
        "discover_entry_point_providers",
        lambda group: [],
    )

    assert validator.main(["--require-plugin"]) == 1


def test_validator_empty_discovery_without_require_plugin_passes(monkeypatch):
    monkeypatch.setattr(
        validator,
        "discover_entry_point_providers",
        lambda group: [],
    )

    assert validator.main([]) == 0


def test_validator_fails_when_discovered_plugin_rejected(monkeypatch):
    monkeypatch.setenv("CORE_PLUGIN_KEY", "secret")
    monkeypatch.setattr(
        validator,
        "discover_entry_point_providers",
        lambda group: [Provider(_descriptor(signature=False))],
    )

    assert validator.main(["--publisher", "core=CORE_PLUGIN_KEY"]) == 1


def test_validator_requires_publisher_key_env(monkeypatch):
    monkeypatch.delenv("CORE_PLUGIN_KEY", raising=False)

    assert validator.main(["--publisher", "core=CORE_PLUGIN_KEY"]) == 2


def test_validator_rejects_invalid_provider_path(monkeypatch):
    monkeypatch.setattr(
        validator,
        "discover_entry_point_providers",
        lambda group: [],
    )

    assert validator.main(["--provider", "not-a-provider"]) == 2
