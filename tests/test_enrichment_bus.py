"""Tests for the Plan D two-tier enrichment plugin bus."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from ocr_local.enrichment_bus import (
    EnrichmentBus,
    ResourceLimits,
    SandboxPolicy,
    UIDescriptor,
    sign_descriptor,
)
from ocr_local.features.custody import EVENT_TYPES


class Provider:
    def __init__(self, descriptor: UIDescriptor) -> None:
        self._descriptor = descriptor

    def ui_descriptor(self) -> UIDescriptor:
        return self._descriptor


def _descriptor(**overrides) -> UIDescriptor:
    descriptor = UIDescriptor(
        plugin_id="sample.redaction",
        display_name="Sample Redaction",
        protocol_version="1.0",
        signature=None,
        resource_limits=ResourceLimits(
            max_memory_mb=128,
            max_cpu_percent=50,
            max_wall_time_ms=1000,
        ),
        deterministic=True,
        network_required=False,
        trust_tier="sandboxed",
        publisher="core",
        publisher_type="core",
    )
    descriptor = replace(descriptor, **overrides)
    return replace(descriptor, signature=sign_descriptor(descriptor, "secret"))


def test_signed_sandboxed_plugin_registers_and_emits_event():
    chain = MagicMock()
    bus = EnrichmentBus(trusted_publishers={"core": "secret"}, custody_chain=chain)

    admission = bus.admit(Provider(_descriptor()))

    assert admission.accepted is True
    assert admission.sandbox_policy_enforced is True
    event_type, payload = chain.log_event.call_args.args
    assert event_type == "PLUGIN_REGISTERED"
    assert payload["tier"] == "sandboxed"
    assert payload["sandbox_policy_enforced"] is True


def test_unsigned_plugin_rejected_and_emits_event():
    chain = MagicMock()
    bus = EnrichmentBus(trusted_publishers={"core": "secret"}, custody_chain=chain)
    descriptor = replace(_descriptor(), signature=None)

    admission = bus.admit(Provider(descriptor))

    assert admission.accepted is False
    assert "signature is required" in admission.reasons
    event_type, payload = chain.log_event.call_args.args
    assert event_type == "PLUGIN_REJECTED"
    assert "signature verification failed" in payload["reasons"]


def test_customer_plugin_cannot_register_in_process():
    bus = EnrichmentBus(trusted_publishers={"vendor": "secret"})
    descriptor = _descriptor(
        publisher="vendor",
        publisher_type="customer",
        trust_tier="in_process",
    )

    admission = bus.admit(Provider(descriptor))

    assert admission.accepted is False
    assert "customer/third-party plugins must use sandboxed tier" in admission.reasons


def test_resource_limit_exhaustion_rejected():
    bus = EnrichmentBus(
        trusted_publishers={"core": "secret"},
        sandbox_policy=SandboxPolicy(max_memory_mb=64),
    )
    descriptor = _descriptor()

    admission = bus.admit(Provider(descriptor))

    assert admission.accepted is False
    assert "resource_limits.max_memory_mb exceeds sandbox policy" in admission.reasons


def test_catalog_exposes_only_registered_plugins():
    bus = EnrichmentBus(trusted_publishers={"core": "secret"})
    bus.admit(Provider(_descriptor(plugin_id="a.plugin", display_name="A")))
    bus.admit(Provider(replace(_descriptor(plugin_id="bad.plugin"), signature=None)))

    catalog = bus.catalog()

    assert [item["plugin_id"] for item in catalog] == ["a.plugin"]
    assert catalog[0]["sandbox_policy_enforced"] is True


def test_plugin_events_registered():
    assert "PLUGIN_REGISTERED" in EVENT_TYPES
    assert "PLUGIN_REJECTED" in EVENT_TYPES
