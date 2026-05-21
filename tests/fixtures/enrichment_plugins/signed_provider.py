"""Signed enrichment plugin fixture for release entry-point validation."""

from __future__ import annotations

from dataclasses import replace

from ocr_local.enrichment_bus import ResourceLimits, UIDescriptor, sign_descriptor


def build_provider():
    class SignedProvider:
        def ui_descriptor(self) -> UIDescriptor:
            descriptor = UIDescriptor(
                plugin_id="fixture.signed_redaction",
                display_name="Fixture Signed Redaction",
                protocol_version="1.0",
                signature=None,
                resource_limits=ResourceLimits(
                    max_memory_mb=64,
                    max_cpu_percent=50,
                    max_wall_time_ms=1000,
                ),
                deterministic=True,
                network_required=False,
                trust_tier="sandboxed",
                publisher="fixture",
                publisher_type="partner",
            )
            return replace(
                descriptor,
                signature=sign_descriptor(descriptor, "fixture-secret"),
            )

    return SignedProvider()
