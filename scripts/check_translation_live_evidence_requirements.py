#!/usr/bin/env python3
"""Check live evidence prerequisites for translation-swarm maintenance review gates.

This script does not run live infrastructure. It makes missing external
evidence explicit and machine-readable so local validation cannot accidentally
claim live TSA, CT2, cluster, or OS sandbox proof.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Requirement:
    key: str
    env_var: str
    description: str
    path_required: bool = False
    artifact_env_var: str | None = None
    artifact_description: str | None = None


REQUIREMENTS = {
    "live-tsa": Requirement(
        key="live-tsa",
        env_var="TRANSLATION_TSA_URL",
        description="reachable RFC3161 TSA endpoint for certified translation evidence",
        artifact_env_var="TRANSLATION_TSA_TSR_PATH",
        artifact_description="stored RFC3161 TSR artifact for dev/live TSA evidence",
    ),
    "ct2-model": Requirement(
        key="ct2-model",
        env_var="TRANSLATION_CT2_MODEL_DIR",
        description="approved CT2 model bundle directory with provenance.json",
        path_required=True,
    ),
    "kubeconfig": Requirement(
        key="kubeconfig",
        env_var="KUBECONFIG",
        description="Kubernetes config for Plan C topology/federation drills",
        path_required=True,
        artifact_env_var="TRANSLATION_CLUSTER_EVIDENCE_PATH",
        artifact_description="captured local or production cluster evidence artifact",
    ),
    "plugin-sandbox": Requirement(
        key="plugin-sandbox",
        env_var="PLUGIN_SANDBOX_RUNTIME_PROOF",
        description="artifact path proving OS process isolation/seccomp/cgroup enforcement",
        path_required=True,
    ),
    "cloud-residency": Requirement(
        key="cloud-residency",
        env_var="TRANSLATION_CLOUD_RESIDENCY_EVIDENCE",
        description="artifact path proving cloud residency, BAA, and zero-retention controls",
        path_required=True,
    ),
}


def check_requirement(req: Requirement) -> tuple[bool, str]:
    if req.artifact_env_var:
        artifact = os.environ.get(req.artifact_env_var, "").strip()
        if artifact:
            if not Path(artifact).exists():
                return False, f"{req.key}: {req.artifact_env_var} path does not exist: {artifact}"
            return True, f"{req.key}: artifact present via {req.artifact_env_var}"

    value = os.environ.get(req.env_var, "").strip()
    if not value:
        return False, f"{req.key}: missing {req.env_var} ({req.description})"
    if req.path_required and not Path(value).exists():
        return False, f"{req.key}: {req.env_var} path does not exist: {value}"
    return True, f"{req.key}: present via {req.env_var}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        action="append",
        choices=sorted(REQUIREMENTS),
        default=[],
        help="require a specific live evidence prerequisite",
    )
    parser.add_argument("--all", action="store_true", help="check all live evidence prerequisites")
    parser.add_argument(
        "--tsa-tsr",
        type=Path,
        default=None,
        help="stored RFC3161 TSR artifact; equivalent to TRANSLATION_TSA_TSR_PATH",
    )
    parser.add_argument(
        "--cluster-evidence",
        type=Path,
        default=None,
        help="captured cluster evidence artifact; equivalent to TRANSLATION_CLUSTER_EVIDENCE_PATH",
    )
    parser.add_argument(
        "--plugin-sandbox-proof",
        type=Path,
        default=None,
        help="captured plugin sandbox proof artifact; equivalent to PLUGIN_SANDBOX_RUNTIME_PROOF",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = sorted(REQUIREMENTS) if args.all else args.require
    if not selected:
        selected = sorted(REQUIREMENTS)
        exit_on_missing = False
    else:
        exit_on_missing = True

    if args.tsa_tsr is not None:
        os.environ["TRANSLATION_TSA_TSR_PATH"] = str(args.tsa_tsr)
    if args.cluster_evidence is not None:
        os.environ["TRANSLATION_CLUSTER_EVIDENCE_PATH"] = str(args.cluster_evidence)
    if args.plugin_sandbox_proof is not None:
        os.environ["PLUGIN_SANDBOX_RUNTIME_PROOF"] = str(args.plugin_sandbox_proof)

    missing = []
    for key in selected:
        ok, message = check_requirement(REQUIREMENTS[key])
        stream = sys.stdout if ok else sys.stderr
        print(message, file=stream)
        if not ok:
            missing.append(key)

    return 1 if missing and exit_on_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
