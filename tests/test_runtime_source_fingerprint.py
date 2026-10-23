from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from runtime_source_fingerprint import fingerprint_for_profile  # noqa: E402


def test_root_cpu_fingerprint_tracks_runtime_source_and_ignores_docs(tmp_path):
    (tmp_path / "api").mkdir()
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\n", encoding="utf-8")
    (tmp_path / ".dockerignore").write_text("docs/\n", encoding="utf-8")
    (tmp_path / "healthcheck.sh").write_text("exit 0\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("fastapi==1\n", encoding="utf-8")
    (tmp_path / "requirements-language-id.txt").write_text(
        "fasttext==1\n", encoding="utf-8"
    )
    (tmp_path / "coordinator").mkdir()
    (tmp_path / "coordinator/requirements.txt").write_text(
        "django==1\n", encoding="utf-8"
    )
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "api/auth.py").write_text("VALUE = 1\n", encoding="utf-8")

    (tmp_path / "coordinator/jobs").mkdir()
    original = fingerprint_for_profile(tmp_path, "root-cpu")
    assert fingerprint_for_profile(tmp_path, "root-cpu") == original

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/note.md").write_text(
        "outside the runtime payload\n", encoding="utf-8"
    )
    assert fingerprint_for_profile(tmp_path, "root-cpu") == original

    generated = tmp_path / "coordinator"
    (generated / "jobs" / "tests").mkdir(parents=True)
    (generated / "jobs" / "tests" / "test_generated.py").write_text(
        "TEST_FIXTURE = True\n", encoding="utf-8"
    )
    (generated / ".pytest_cache" / "v" / "cache").mkdir(parents=True)
    (generated / ".pytest_cache" / "v" / "cache" / "nodeids").write_text(
        "[]\n", encoding="utf-8"
    )
    (generated / "failover_drill_results").mkdir()
    (generated / "failover_drill_results" / "result.json").write_text(
        "{}\n", encoding="utf-8"
    )
    assert fingerprint_for_profile(tmp_path, "root-cpu") == original

    (tmp_path / "api/auth.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert fingerprint_for_profile(tmp_path, "root-cpu") != original


def test_coordinator_fingerprint_tracks_coordinator_source(tmp_path):
    (tmp_path / "coordinator").mkdir()
    (tmp_path / "ocr_distributed").mkdir()
    (tmp_path / "ocr_local").mkdir()
    (tmp_path / ".dockerignore").write_text(
        "**/tests/\n**/.pytest_cache/\n**/failover_drill_results/\n",
        encoding="utf-8",
    )
    (tmp_path / "coordinator/Dockerfile.coordinator").write_text(
        "FROM python:3.12\n", encoding="utf-8"
    )
    (tmp_path / "coordinator/requirements.txt").write_text(
        "django==1\n", encoding="utf-8"
    )
    (tmp_path / "coordinator/jobs.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "ocr_distributed/task.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "ocr_local/policy.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "optimize_pdfs.py").write_text("VALUE = 1\n", encoding="utf-8")

    (tmp_path / "coordinator/jobs").mkdir()
    original = fingerprint_for_profile(tmp_path, "coordinator")
    scratch = tmp_path / "coordinator" / "_test_nfs_root"
    scratch.mkdir()
    (scratch / "fixture.bin").write_bytes(b"ignored by docker context")
    (tmp_path / "coordinator" / ".env").write_text(
        "LOCAL_SECRET=not-runtime-input\n", encoding="utf-8"
    )
    (tmp_path / "coordinator" / "debug.log").write_text(
        "generated output\n", encoding="utf-8"
    )
    assert fingerprint_for_profile(tmp_path, "coordinator") == original

    (tmp_path / "coordinator/jobs/tests").mkdir(parents=True)
    (tmp_path / "coordinator/jobs/tests/test_source.py").write_text(
        "TEST_FIXTURE = True\n", encoding="utf-8"
    )
    (tmp_path / "coordinator/.pytest_cache/v/cache").mkdir(parents=True)
    (tmp_path / "coordinator/.pytest_cache/v/cache/nodeids").write_text(
        "[]\n", encoding="utf-8"
    )
    (tmp_path / "coordinator/failover_drill_results").mkdir()
    (tmp_path / "coordinator/failover_drill_results/result.json").write_text(
        "{}\n", encoding="utf-8"
    )
    assert fingerprint_for_profile(tmp_path, "coordinator") == original

    (tmp_path / "coordinator/jobs.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert fingerprint_for_profile(tmp_path, "coordinator") != original


def test_frontend_fingerprint_tracks_built_source_and_ignores_node_artifacts(tmp_path):
    (tmp_path / "frontend/src/app").mkdir(parents=True)
    (tmp_path / "frontend/public").mkdir()
    (tmp_path / "frontend/package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "frontend/package-lock.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "Dockerfile.frontend").write_text("FROM node:26\n", encoding="utf-8")
    (tmp_path / "Dockerfile.frontend.dockerignore").write_text(
        "**\n!frontend/src/**\n", encoding="utf-8"
    )
    (tmp_path / ".dockerignore").write_text("frontend/\n", encoding="utf-8")
    (tmp_path / "frontend/src/app/page.tsx").write_text(
        "export default 1\n", encoding="utf-8"
    )
    original = fingerprint_for_profile(tmp_path, "frontend")

    (tmp_path / "frontend/node_modules/pkg").mkdir(parents=True)
    (tmp_path / "frontend/node_modules/pkg/index.js").write_text(
        "generated\n", encoding="utf-8"
    )
    (tmp_path / "frontend/.next/cache").mkdir(parents=True)
    (tmp_path / "frontend/.next/cache/build").write_text("generated", encoding="utf-8")
    assert fingerprint_for_profile(tmp_path, "frontend") == original

    (tmp_path / "frontend/src/app/page.tsx").write_text(
        "export default 2\n", encoding="utf-8"
    )
    assert fingerprint_for_profile(tmp_path, "frontend") != original


@pytest.mark.parametrize("profile", ["frontend", "worker"])
@pytest.mark.parametrize("directory", ["dashboard", "docs", "models", "sdk"])
def test_nested_app_source_not_confused_with_root_workspace(
    tmp_path, profile, directory,
):
    source = tmp_path / "frontend" / "src" / "app" / directory / "page.tsx"
    source.parent.mkdir(parents=True)
    source.write_text("export default 1\n", encoding="utf-8")
    workspace = tmp_path / directory
    workspace.mkdir()
    note = workspace / "note.md"
    note.write_text("operator workspace\n", encoding="utf-8")
    baseline = fingerprint_for_profile(tmp_path, profile)

    note.write_text("changed operator workspace\n", encoding="utf-8")
    assert fingerprint_for_profile(tmp_path, profile) == baseline

    source.write_text("export default 2\n", encoding="utf-8")
    assert fingerprint_for_profile(tmp_path, profile) != baseline


def test_worker_fingerprint_tracks_worker_dockerfile_and_runtime_source(tmp_path):
    for relative in (
        "api",
        "coordinator",
        "ocr_distributed",
        "ocr_local",
        "reprocess",
        "schemas",
        "security",
    ):
        (tmp_path / relative).mkdir()
    (tmp_path / ".dockerignore").write_text("docs/\n", encoding="utf-8")
    (tmp_path / "coordinator/Dockerfile.worker.gpu").write_text(
        "FROM cuda\n", encoding="utf-8"
    )
    original = fingerprint_for_profile(tmp_path, "worker")
    (tmp_path / "coordinator/Dockerfile.worker.gpu").write_text(
        "FROM cuda:patched\n", encoding="utf-8"
    )
    after_dockerfile = fingerprint_for_profile(tmp_path, "worker")
    assert after_dockerfile != original

    (tmp_path / "config/profiles").mkdir(parents=True)
    (tmp_path / "config/profiles/solo.json").write_text(
        '{"workers": 8}\n', encoding="utf-8"
    )
    after_config = fingerprint_for_profile(tmp_path, "worker")
    assert after_config != after_dockerfile

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/review.md").write_text("not sent to Docker", encoding="utf-8")
    (tmp_path / "coordinator/tests").mkdir()
    (tmp_path / "coordinator/tests/test_review.py").write_text(
        "assert True", encoding="utf-8"
    )
    assert fingerprint_for_profile(tmp_path, "worker") == after_config

    ruff_cache = tmp_path / ".ruff_cache"
    ruff_cache.mkdir()
    cache_file = ruff_cache / "cache-key"
    cache_file.write_text("generated cache", encoding="utf-8")
    with_cache = fingerprint_for_profile(tmp_path, "worker")
    cache_file.write_text("changed generated cache", encoding="utf-8")
    assert fingerprint_for_profile(tmp_path, "worker") == with_cache


@pytest.mark.parametrize("explicit_revision", [None, "a" * 40])
def test_archive_wrapper_does_not_bind_enclosing_checkout(tmp_path, explicit_revision):
    parent = tmp_path / "enclosing"
    parent.mkdir()
    subprocess.run(["git", "init", "-q", str(parent)], check=True)
    (parent / "app.py").write_text("WRONG_SOURCE = True\n", encoding="utf-8")
    root = parent / "extracted"
    (root / "scripts").mkdir(parents=True)
    for name in ("with_runtime_source.sh", "runtime_source_fingerprint.py"):
        shutil.copy2(SCRIPTS / name, root / "scripts" / name)
    (root / "app.py").write_text("ARCHIVE_SOURCE = True\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.pop("SOURCE_REVISION", None)
    if explicit_revision is not None:
        env["SOURCE_REVISION"] = explicit_revision
    result = subprocess.run(
        [
            "bash", str(root / "scripts/with_runtime_source.sh"), "root-cpu",
            sys.executable, "-c",
            "import os,json; print(json.dumps([os.environ['SOURCE_REVISION'], "
            "os.environ['SOURCE_TREE_SHA256']]))",
        ],
        cwd=parent, env=env, check=True, capture_output=True, text=True,
    )
    revision, digest = json.loads(result.stdout)
    expected = fingerprint_for_profile(root, "root-cpu")
    assert digest == expected
    assert digest != fingerprint_for_profile(parent, "root-cpu")
    assert revision == (explicit_revision or expected)


def test_checkout_wrapper_uses_own_revision_from_foreign_working_directory(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    for name in ("with_runtime_source.sh", "runtime_source_fingerprint.py"):
        shutil.copy2(SCRIPTS / name, root / "scripts" / name)
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["SOURCE_REVISION"] = "a" * 40
    subprocess.run(["git", "init", "-q", str(root)], env=env, check=True)
    subprocess.run(
        [
            "git", "-C", str(root), "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
            "commit", "--allow-empty", "-qm", "fixture",
        ],
        env=env, check=True,
    )
    expected_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], env=env, text=True,
    ).strip()
    result = subprocess.run(
        [
            "bash", str(root / "scripts/with_runtime_source.sh"), "root-cpu",
            sys.executable, "-c",
            "import os,json; print(json.dumps([os.environ['SOURCE_REVISION'], "
            "os.environ['SOURCE_TREE_SHA256']]))",
        ],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == [
        expected_revision, fingerprint_for_profile(root, "root-cpu"),
    ]
