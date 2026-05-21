#!/usr/bin/env python3
"""Validate installed Plan D enrichment entry-point plugins."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_local.enrichment_bus import EnrichmentBus, discover_entry_point_providers  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher", action="append", default=[])
    parser.add_argument("--group", default="ocr_local.enrichments")
    parser.add_argument(
        "--require-plugin",
        action="store_true",
        help="fail when no entry-point plugins are discovered",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="additional provider factory import path, formatted module:function",
    )
    return parser.parse_args(argv)


def _load_provider_factory(import_path: str):
    module_name, sep, attr = import_path.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"invalid --provider {import_path!r}; expected module:function"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trusted: dict[str, str] = {}
    for item in args.publisher:
        name, sep, env_name = item.partition("=")
        if not sep:
            print(f"invalid --publisher {item!r}; expected name=ENV_VAR", file=sys.stderr)
            return 2
        key = os.environ.get(env_name)
        if not key:
            print(f"trusted publisher key env var is unset: {env_name}", file=sys.stderr)
            return 2
        trusted[name] = key

    bus = EnrichmentBus(trusted_publishers=trusted)
    failures = []
    providers = discover_entry_point_providers(args.group)
    for import_path in args.provider:
        try:
            providers.append(_load_provider_factory(import_path))
        except (ImportError, AttributeError, ValueError) as exc:
            print(f"invalid provider {import_path!r}: {exc}", file=sys.stderr)
            return 2
    if args.require_plugin and not providers:
        print(
            f"enrichment plugin validation failed: no plugins discovered for group {args.group!r}",
            file=sys.stderr,
        )
        return 1
    for provider in providers:
        admission = bus.admit(provider)
        if not admission.accepted:
            failures.append((admission.descriptor.plugin_id, admission.reasons))

    if failures:
        print("enrichment plugin validation failed:", file=sys.stderr)
        for plugin_id, reasons in failures:
            print(f"- {plugin_id}: {', '.join(reasons)}", file=sys.stderr)
        return 1
    print("enrichment plugin validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
