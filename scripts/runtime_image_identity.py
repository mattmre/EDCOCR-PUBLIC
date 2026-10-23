"""Capture and verify immutable image identity for a running Kubernetes pod."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")


def _canonical_image_id(value: str) -> str:
    candidate = value.rsplit("@", 1)[-1]
    if candidate.startswith("docker://") or candidate.startswith("containerd://"):
        candidate = candidate.split("://", 1)[1]
    return candidate


def inspect_docker_image(image_id: str) -> dict[str, Any]:
    """Read labels and platform from the host image matching a live Pod ID."""
    canonical_id = _canonical_image_id(image_id)
    if not SHA256_RE.fullmatch(canonical_id):
        raise ValueError(f"live image ID is not an immutable sha256 ID: {image_id!r}")
    result = subprocess.run(
        ["docker", "image", "inspect", canonical_id],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker cannot inspect the running image {canonical_id}: "
            f"{(result.stderr or '').strip()[-600:]}"
        )
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("docker image inspect returned invalid JSON") from exc
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("docker image inspect did not return one image")
    image = rows[0]
    actual_id = str(image.get("Id", ""))
    if actual_id != canonical_id:
        raise RuntimeError(
            f"Docker image ID {actual_id!r} does not match the running Pod ID"
        )
    config = image.get("Config") if isinstance(image.get("Config"), dict) else {}
    labels = config.get("Labels") if isinstance(config.get("Labels"), dict) else {}
    return {
        "docker_image_id": actual_id,
        "platform": f"{image.get('Os', '')}/{image.get('Architecture', '')}",
        "source_revision": labels.get("org.opencontainers.image.revision"),
        "source_tree_sha256": labels.get("io.ocr_local.runtime-source-sha256"),
        "source_profile": labels.get("io.ocr_local.runtime-profile"),
        "repo_digests": image.get("RepoDigests") or [],
    }


def pod_container_identity(pod: dict[str, Any], container_name: str) -> dict[str, Any]:
    """Extract the scheduled image and runtime ID from one Pod status object."""
    metadata = pod.get("metadata") if isinstance(pod.get("metadata"), dict) else {}
    spec = pod.get("spec") if isinstance(pod.get("spec"), dict) else {}
    status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
    labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
    containers = (
        spec.get("containers") if isinstance(spec.get("containers"), list) else []
    )
    container = next(
        (
            item
            for item in containers
            if isinstance(item, dict) and item.get("name") == container_name
        ),
        None,
    )
    statuses = status.get("containerStatuses")
    statuses = statuses if isinstance(statuses, list) else []
    container_status = next(
        (
            item
            for item in statuses
            if isinstance(item, dict) and item.get("name") == container_name
        ),
        None,
    )
    if container is None or container_status is None:
        raise ValueError(f"Pod lacks container or status for {container_name!r}")
    image_id = str(container_status.get("imageID", ""))
    container_id = str(container_status.get("containerID", ""))
    canonical_image_id = _canonical_image_id(image_id)
    if not SHA256_RE.fullmatch(canonical_image_id):
        raise ValueError(f"Pod has no immutable sha256 image ID: {image_id!r}")
    if not container_id:
        raise ValueError("Pod container status has no container ID")
    return {
        "namespace": metadata.get("namespace"),
        "pod": metadata.get("name"),
        "pod_uid": metadata.get("uid"),
        "node": spec.get("nodeName"),
        "component": labels.get("app.kubernetes.io/component"),
        "container": container_name,
        "declared_image": container.get("image"),
        "image_id": image_id,
        "container_id": container_id,
        "ready": container_status.get("ready") is True,
        "restart_count": container_status.get("restartCount"),
    }


def verified_pod_container_identity(
    pod: dict[str, Any],
    container_name: str,
    *,
    expected_source_tree_sha256: str,
    expected_source_profile: str,
) -> dict[str, Any]:
    """Bind a Pod's live image ID to the labeled local image and source tree."""
    identity = pod_container_identity(pod, container_name)
    image = inspect_docker_image(identity["image_id"])
    if image["source_tree_sha256"] != expected_source_tree_sha256:
        raise ValueError(
            "running image source fingerprint does not match this checkout: "
            f"expected {expected_source_tree_sha256}, "
            f"got {image['source_tree_sha256']!r}"
        )
    if image["source_profile"] != expected_source_profile:
        raise ValueError(
            f"running image source profile must be {expected_source_profile!r}"
        )
    revision = image["source_revision"]
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise ValueError("running image has no full Git source revision label")
    identity.update(image)
    return identity
