"""Plan D two-tier enrichment plugin bus admission core.

This module enforces descriptor, signature, tier, and resource-limit policy.
It does not provide OS/process sandboxing; seccomp/cgroup runtime proof remains
a separate deployment-gate requirement.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

TrustTier = Literal["in_process", "sandboxed"]
PublisherType = Literal["core", "partner", "customer", "third_party"]


@dataclass(frozen=True)
class ResourceLimits:
    max_memory_mb: int
    max_cpu_percent: int
    max_wall_time_ms: int


@dataclass(frozen=True)
class UIDescriptor:
    plugin_id: str
    display_name: str
    protocol_version: str
    signature: str | None
    resource_limits: ResourceLimits | None
    deterministic: bool
    network_required: bool
    trust_tier: TrustTier = "sandboxed"
    publisher: str = "unknown"
    publisher_type: PublisherType = "third_party"


class EnrichmentProvider(Protocol):
    def ui_descriptor(self) -> UIDescriptor:
        """Return the provider descriptor used for admission/catalog."""


@dataclass(frozen=True)
class SandboxPolicy:
    available: bool = True
    max_memory_mb: int = 1024
    max_cpu_percent: int = 100
    max_wall_time_ms: int = 60_000

    def validate(self, limits: ResourceLimits | None) -> list[str]:
        if limits is None:
            return ["resource_limits is required"]
        problems: list[str] = []
        if not self.available:
            problems.append("sandbox policy enforcement unavailable")
        if limits.max_memory_mb <= 0 or limits.max_memory_mb > self.max_memory_mb:
            problems.append("resource_limits.max_memory_mb exceeds sandbox policy")
        if limits.max_cpu_percent <= 0 or limits.max_cpu_percent > self.max_cpu_percent:
            problems.append("resource_limits.max_cpu_percent exceeds sandbox policy")
        if limits.max_wall_time_ms <= 0 or limits.max_wall_time_ms > self.max_wall_time_ms:
            problems.append("resource_limits.max_wall_time_ms exceeds sandbox policy")
        return problems


@dataclass(frozen=True)
class PluginAdmission:
    descriptor: UIDescriptor
    accepted: bool
    reasons: tuple[str, ...]
    sandbox_policy_enforced: bool = False


def _signature_payload(descriptor: UIDescriptor) -> bytes:
    data = asdict(descriptor)
    data["signature"] = None
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_descriptor(descriptor: UIDescriptor, key: str) -> str:
    return hmac.new(key.encode("utf-8"), _signature_payload(descriptor), hashlib.sha256).hexdigest()


def verify_descriptor_signature(
    descriptor: UIDescriptor,
    trusted_publishers: dict[str, str],
) -> bool:
    if not descriptor.signature:
        return False
    key = trusted_publishers.get(descriptor.publisher)
    if not key:
        return False
    expected = sign_descriptor(descriptor, key)
    return hmac.compare_digest(expected, descriptor.signature)


class EnrichmentBus:
    def __init__(
        self,
        *,
        trusted_publishers: dict[str, str],
        custody_chain: Any | None = None,
        sandbox_policy: SandboxPolicy | None = None,
    ) -> None:
        self.trusted_publishers = trusted_publishers
        self.custody_chain = custody_chain
        self.sandbox_policy = sandbox_policy or SandboxPolicy()
        self._plugins: dict[str, PluginAdmission] = {}

    def admit(self, provider: EnrichmentProvider) -> PluginAdmission:
        descriptor = provider.ui_descriptor()
        reasons = self._validate_descriptor(descriptor)
        accepted = not reasons
        admission = PluginAdmission(
            descriptor=descriptor,
            accepted=accepted,
            reasons=tuple(reasons),
            sandbox_policy_enforced=accepted and descriptor.trust_tier == "sandboxed",
        )
        if accepted:
            self._plugins[descriptor.plugin_id] = admission
            self._emit(
                "PLUGIN_REGISTERED",
                {
                    "plugin_id": descriptor.plugin_id,
                    "tier": descriptor.trust_tier,
                    "publisher": descriptor.publisher,
                    "sandbox_policy_enforced": admission.sandbox_policy_enforced,
                },
            )
        else:
            self._emit(
                "PLUGIN_REJECTED",
                {
                    "plugin_id": descriptor.plugin_id,
                    "tier": descriptor.trust_tier,
                    "publisher": descriptor.publisher,
                    "reasons": list(admission.reasons),
                },
            )
        return admission

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "plugin_id": item.descriptor.plugin_id,
                "display_name": item.descriptor.display_name,
                "protocol_version": item.descriptor.protocol_version,
                "tier": item.descriptor.trust_tier,
                "publisher": item.descriptor.publisher,
                "deterministic": item.descriptor.deterministic,
                "network_required": item.descriptor.network_required,
                "sandbox_policy_enforced": item.sandbox_policy_enforced,
            }
            for item in sorted(self._plugins.values(), key=lambda p: p.descriptor.plugin_id)
        ]

    def _validate_descriptor(self, descriptor: UIDescriptor) -> list[str]:
        problems: list[str] = []
        if not descriptor.protocol_version:
            problems.append("protocol_version is required")
        if descriptor.resource_limits is None:
            problems.append("resource_limits is required")
        if descriptor.signature is None:
            problems.append("signature is required")
        if descriptor.publisher_type in {"customer", "third_party"} and descriptor.trust_tier != "sandboxed":
            problems.append("customer/third-party plugins must use sandboxed tier")
        if not verify_descriptor_signature(descriptor, self.trusted_publishers):
            problems.append("signature verification failed")
        if descriptor.trust_tier == "sandboxed":
            problems.extend(self.sandbox_policy.validate(descriptor.resource_limits))
        return problems

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.custody_chain is not None:
            self.custody_chain.log_event(event_type, payload)


def discover_entry_point_providers(group: str = "ocr_local.enrichments") -> list[EnrichmentProvider]:
    entry_points = importlib.metadata.entry_points()
    selected = entry_points.select(group=group)
    return [entry_point.load()() for entry_point in selected]
