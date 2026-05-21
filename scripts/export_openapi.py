#!/usr/bin/env python3
"""Export OpenAPI spec from the FastAPI application.

Usage:
    python scripts/export_openapi.py                    # writes docs/openapi.json
    python scripts/export_openapi.py --output api.json  # custom output path
    python scripts/export_openapi.py --yaml             # YAML format (requires pyyaml)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Enable all feature gates so the spec includes every endpoint.
os.environ.setdefault("ENABLE_MULTITENANCY", "true")
os.environ.setdefault("ENABLE_DASHBOARD", "true")
os.environ.setdefault("OCR_API_KEY", "export-dummy-key")
os.environ.setdefault("OCR_OUTPUT_DIR", "/tmp/ocr_output")

# Add project root to path so ``api`` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAPI spec")
    parser.add_argument(
        "--output",
        "-o",
        default="docs/openapi.json",
        help="Output file path (default: docs/openapi.json)",
    )
    parser.add_argument(
        "--yaml",
        action="store_true",
        help="Output in YAML format (requires pyyaml)",
    )
    args = parser.parse_args()

    app = create_app()
    schema = app.openapi()

    # Print summary
    paths = schema.get("paths", {})
    endpoint_count = sum(len(methods) for methods in paths.values())
    print(f"OpenAPI spec: {schema['info']['title']} v{schema['info']['version']}")
    print(f"Endpoints: {endpoint_count} across {len(paths)} paths")
    print(f"Tags: {len(schema.get('tags', []))}")
    security_schemes = list(
        schema.get("components", {}).get("securitySchemes", {}).keys()
    )
    print(f"Security schemes: {security_schemes}")

    output_path: str = args.output
    if args.yaml:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            print(
                "ERROR: pyyaml not installed. Install with: pip install pyyaml",
                file=sys.stderr,
            )
            sys.exit(1)
        if not output_path.endswith((".yml", ".yaml")):
            output_path = output_path.rsplit(".", 1)[0] + ".yaml"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                schema,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
            f.write("\n")

    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
