#!/usr/bin/env python3
"""Remove inactive OpenCV payloads and install fail-closed OCR controls."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import site
import subprocess
from pathlib import Path


class HardeningError(RuntimeError):
    """Raised when the installed worker dependency shape is unexpected."""


def _site_packages() -> Path:
    candidates = [Path(item) for item in site.getsitepackages()]
    existing = [item for item in candidates if item.is_dir()]
    if len(existing) != 1:
        raise HardeningError(f"expected one site-packages directory, found {existing}")
    return existing[0]


def _cv2_extension(site_packages: Path) -> Path:
    matches = list((site_packages / "cv2").glob("cv2*.so"))
    if len(matches) != 1:
        raise HardeningError(f"expected one cv2 extension, found {matches}")
    return matches[0]


def _active_opencv_payloads(site_packages: Path, extension: Path) -> set[Path]:
    completed = subprocess.run(
        ["ldd", str(extension)],
        check=True,
        capture_output=True,
        text=True,
    )
    active: set[Path] = set()
    for line in completed.stdout.splitlines():
        if "=>" not in line:
            continue
        resolved = line.split("=>", 1)[1].strip().split(" ", 1)[0]
        path = Path(resolved)
        if (
            path.is_absolute()
            and path.parent.name.startswith("opencv_")
            and path.parent.name.endswith(".libs")
        ):
            active.add(path.parent.resolve())
    if len(active) != 1:
        raise HardeningError(
            f"expected one linked OpenCV payload, found {sorted(active)}"
        )
    if not next(iter(active)).is_relative_to(site_packages.resolve()):
        raise HardeningError("linked OpenCV payload is outside site-packages")
    return active


def harden(template: Path) -> dict[str, object]:
    site_packages = _site_packages()
    extension = _cv2_extension(site_packages)
    active = _active_opencv_payloads(site_packages, extension)
    payloads = sorted(site_packages.glob("opencv_*.libs"))
    inactive = [item for item in payloads if item.resolve() not in active]
    if not inactive:
        raise HardeningError("expected at least one inactive OpenCV payload to remove")
    for directory in inactive:
        shutil.rmtree(directory)

    target = site_packages / "sitecustomize.py"
    if target.exists() and target.read_bytes() != template.read_bytes():
        raise HardeningError(f"refusing to overwrite unrelated {target}")
    shutil.copyfile(template, target)
    return {
        "activePayload": str(next(iter(active))),
        "removedPayloads": [str(item) for item in inactive],
        "sitecustomize": str(target),
    }


def verify() -> None:
    site_packages = _site_packages()
    extension = _cv2_extension(site_packages)
    active = _active_opencv_payloads(site_packages, extension)
    payloads = {item.resolve() for item in site_packages.glob("opencv_*.libs")}
    if payloads != active:
        raise HardeningError(
            "OpenCV payload inventory does not equal linked inventory: "
            f"{sorted(payloads)}"
        )
    target = site_packages / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("ocr_worker_sitecustomize", target)
    if spec is None or spec.loader is None:
        raise HardeningError("worker sitecustomize is not importable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        print("OCR_WORKER_RUNTIME_HARDENING_VERIFY=PASS")
        return 0
    if args.template is None or not args.template.is_file():
        parser.error("--template must name the committed sitecustomize source")
    evidence = harden(args.template)
    print(f"OCR_WORKER_RUNTIME_HARDENING=PASS evidence={evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
