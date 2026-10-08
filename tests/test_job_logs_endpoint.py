"""Tests for D7 per-job NDJSON log endpoint and writer.

Covers:
- ``write_job_log`` round-trip + concurrent writers (4 threads)
- ``GET /api/v1/jobs/{job_id}/logs`` 200 NDJSON response
- 404 ``NO_PER_JOB_LOGS`` when the file is missing
- 404 (not 403) on cross-tenant access
- ``since=`` / ``limit=`` / ``level=`` filtering
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    from api import auth as auth_mod
    from api import config as config_mod

    monkeypatch.setenv("OCR_API_KEY", "test-key-d7")
    monkeypatch.setenv("OCR_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("OCR_SOURCE_DIR", str(tmp_path / "source"))
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "false")
    monkeypatch.setattr(config_mod, "OCR_API_KEY", "test-key-d7")
    monkeypatch.setattr(config_mod, "ALLOW_UNAUTHENTICATED", False)
    monkeypatch.setattr(auth_mod, "OCR_API_KEY", "test-key-d7")
    monkeypatch.setattr(auth_mod, "ALLOW_UNAUTHENTICATED", False)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "source").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from api.main import create_app

    app = create_app()
    app.state.limiter.enabled = False
    app.state.limiter.reset()
    test_client = TestClient(app)
    test_client.headers.update({"X-API-Key": "test-key-d7"})
    return test_client


def _seed_log(tmp_path: Path, job_id: str, records):
    from api.job_log_writer import job_log_path

    path = job_log_path(job_id, base_dir=str(tmp_path / "output"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# Writer-level tests
# ---------------------------------------------------------------------------


class TestWriter:
    def test_round_trip(self, _setup):
        from api.job_log_writer import job_log_path, parse_log_line, write_job_log

        write_job_log(
            "job_aaaaaaaa1111",
            {"level": "INFO", "code": "JOB_STARTED", "message": "hi"},
            base_dir=str(_setup / "output"),
        )
        path = job_log_path("job_aaaaaaaa1111", base_dir=str(_setup / "output"))
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = parse_log_line(lines[0])
        assert rec["level"] == "INFO"
        assert rec["code"] == "JOB_STARTED"
        assert rec["message"] == "hi"
        assert rec["job_id"] == "job_aaaaaaaa1111"
        # Auto-stamped ts is ISO-8601.
        datetime.fromisoformat(rec["ts"])

    def test_concurrent_writers(self, _setup):
        """4 threads writing 25 records each must yield 100 valid JSON lines."""
        from api.job_log_writer import job_log_path, write_job_log

        job_id = "job_aaaaaaaa2222"
        base_dir = str(_setup / "output")

        def _writer(prefix: str):
            for i in range(25):
                write_job_log(
                    job_id,
                    {"level": "INFO", "code": "TICK", "message": f"{prefix}-{i}"},
                    base_dir=base_dir,
                )

        threads = [
            threading.Thread(target=_writer, args=(f"t{i}",), daemon=True)
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        path = job_log_path(job_id, base_dir=base_dir)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 100
        # Every line must parse; no torn writes.
        for line in lines:
            json.loads(line)

    def test_invalid_level_normalised(self, _setup):
        from api.job_log_writer import job_log_path, parse_log_line, write_job_log

        write_job_log(
            "job_aaaaaaaa3333",
            {"level": "BANANA", "message": "bad level"},
            base_dir=str(_setup / "output"),
        )
        path = job_log_path("job_aaaaaaaa3333", base_dir=str(_setup / "output"))
        rec = parse_log_line(path.read_text(encoding="utf-8").splitlines()[0])
        assert rec["level"] == "INFO"

    def test_oversized_record_is_not_written(self, _setup):
        from api.job_log_writer import (
            _MAX_LOG_RECORD_BYTES,
            job_log_path,
            write_job_log,
        )

        job_id = "job_aaaaaaaa7777"
        write_job_log(
            job_id,
            {"message": "x" * (_MAX_LOG_RECORD_BYTES + 1)},
            base_dir=str(_setup / "output"),
        )

        assert not job_log_path(job_id, base_dir=str(_setup / "output")).exists()


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestEndpoint404:
    def test_missing_file_returns_404_with_code(self, client):
        resp = client.get("/api/v1/jobs/job_aaaaaaaa9999/logs")
        assert resp.status_code == 404
        body = resp.json()
        # FastAPI wraps dict detail under "detail".
        assert body["detail"]["code"] == "NO_PER_JOB_LOGS"

    def test_invalid_job_id_returns_400(self, client):
        resp = client.get("/api/v1/jobs/not-a-job-id/logs")
        assert resp.status_code == 400


class TestEndpoint200:
    def test_reads_logs_written_at_the_pipeline_job_output_path(self, client, _setup):
        from api.job_log_writer import write_job_log

        job_id = "job_aaaaaaaa4321"
        write_job_log(job_id, {"code": "JOB_STARTED", "message": "pipeline"},
                      base_dir=str(_setup / "output" / job_id))
        response = client.get(f"/api/v1/jobs/{job_id}/logs")
        assert response.status_code == 200
        assert json.loads(response.text)["code"] == "JOB_STARTED"

    def test_job_directory_symlink_cannot_redirect_logs(self, client, _setup):
        from api.job_log_writer import write_job_log

        own_id, foreign_id = "job_aaaaaaaa1234", "job_bbbbbbbb1234"
        foreign = _setup / "output" / foreign_id
        write_job_log(own_id, {"message": "foreign bytes"}, base_dir=str(foreign))
        (_setup / "output" / own_id).symlink_to(foreign, target_is_directory=True)
        response = client.get(f"/api/v1/jobs/{own_id}/logs")
        assert response.status_code == 404
        assert "foreign bytes" not in response.text

    def test_output_variable_precedence_matches_api_config(self, _setup, monkeypatch):
        from api.job_log_writer import job_log_path

        monkeypatch.setenv("OUTPUT_FOLDER", str(_setup / "canonical"))
        monkeypatch.setenv("OCR_OUTPUT_DIR", str(_setup / "legacy"))
        assert job_log_path("job_aaaaaaaa1234").is_relative_to(_setup / "canonical")

    def test_skips_oversized_line_without_hiding_following_record(self, client, _setup):
        from api.job_log_writer import _MAX_LOG_RECORD_BYTES

        path = _seed_log(
            _setup,
            "job_aaaaaaaa8888",
            [
                {
                    "job_id": "job_aaaaaaaa8888",
                    "level": "INFO",
                    "code": "OK",
                    "message": "kept",
                }
            ],
        )
        existing = path.read_text(encoding="utf-8")
        path.write_text(
            "x" * (_MAX_LOG_RECORD_BYTES + 1) + "\n" + existing,
            encoding="utf-8",
        )

        resp = client.get("/api/v1/jobs/job_aaaaaaaa8888/logs")

        assert resp.status_code == 200
        assert [json.loads(line)["code"] for line in resp.text.splitlines()] == ["OK"]

    def test_streams_ndjson(self, client, _setup):
        records = [
            {
                "ts": "2026-04-27T12:00:00+00:00",
                "level": "INFO",
                "code": "JOB_STARTED",
                "job_id": "job_aaaaaaaa4444",
                "message": "hello",
            },
            {
                "ts": "2026-04-27T12:00:01+00:00",
                "level": "ERROR",
                "code": "JOB_FAILED",
                "job_id": "job_aaaaaaaa4444",
                "message": "boom",
            },
        ]
        _seed_log(_setup, "job_aaaaaaaa4444", records)

        resp = client.get("/api/v1/jobs/job_aaaaaaaa4444/logs")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = resp.text.strip().splitlines()
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["code"] == "JOB_STARTED"
        assert parsed[1]["code"] == "JOB_FAILED"

    def test_since_filter(self, client, _setup):
        base = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            {
                "ts": (base + timedelta(seconds=i)).isoformat(),
                "level": "INFO",
                "code": "TICK",
                "job_id": "job_aaaaaaaa5555",
                "message": f"#{i}",
            }
            for i in range(5)
        ]
        _seed_log(_setup, "job_aaaaaaaa5555", records)

        cutoff = (base + timedelta(seconds=2)).isoformat()
        # Pass via params= so urllib quotes the timezone offset properly.
        resp = client.get(
            "/api/v1/jobs/job_aaaaaaaa5555/logs",
            params={"since": cutoff},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().splitlines()
        # Strictly after cutoff -> seconds 3, 4 only.
        assert len(lines) == 2

    def test_level_filter(self, client, _setup):
        records = [
            {
                "ts": "2026-04-27T12:00:00+00:00",
                "level": "DEBUG",
                "code": "X",
                "job_id": "job_aaaaaaaa6666",
                "message": "d",
            },
            {
                "ts": "2026-04-27T12:00:01+00:00",
                "level": "INFO",
                "code": "X",
                "job_id": "job_aaaaaaaa6666",
                "message": "i",
            },
            {
                "ts": "2026-04-27T12:00:02+00:00",
                "level": "ERROR",
                "code": "X",
                "job_id": "job_aaaaaaaa6666",
                "message": "e",
            },
        ]
        _seed_log(_setup, "job_aaaaaaaa6666", records)
        resp = client.get("/api/v1/jobs/job_aaaaaaaa6666/logs?level=WARN")
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.text.strip().splitlines()]
        # Only ERROR satisfies level >= WARN.
        assert len(lines) == 1
        assert lines[0]["level"] == "ERROR"

    def test_limit_caps_records(self, client, _setup):
        records = [
            {
                "ts": f"2026-04-27T12:00:{i:02d}+00:00",
                "level": "INFO",
                "code": "TICK",
                "job_id": "job_aaaaaaaa7777",
                "message": str(i),
            }
            for i in range(10)
        ]
        _seed_log(_setup, "job_aaaaaaaa7777", records)
        resp = client.get("/api/v1/jobs/job_aaaaaaaa7777/logs?limit=3")
        assert resp.status_code == 200
        lines = resp.text.strip().splitlines()
        assert len(lines) == 3

    def test_cursor_paginates_colliding_timestamps_and_appends(self, client, _setup):
        job_id = "job_aaaaaaaa7171"
        records = [
            {"ts": "2026-04-27T12:00:00+00:00", "level": "INFO", "code": "TICK", "job_id": job_id, "message": str(i)}
            for i in range(5)
        ]
        path = _seed_log(_setup, job_id, records)
        url = f"/api/v1/jobs/{job_id}/logs"
        cursor = None
        seen = []
        for _ in range(4):
            response = client.get(url, params={"limit": 2, **({"cursor": cursor} if cursor else {})})
            assert response.status_code == 200
            seen.extend(json.loads(line)["message"] for line in response.text.splitlines())
            cursor = response.headers["x-next-cursor"]
        assert seen == ["0", "1", "2", "3", "4"]

        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**records[0], "message": "5"}) + "\n")
        response = client.get(url, params={"cursor": cursor})
        assert [json.loads(line)["message"] for line in response.text.splitlines()] == ["5"]

    def test_cursor_resets_after_replacement_and_never_accepts_a_path(self, client, _setup):
        job_id = "job_aaaaaaaa7272"
        url = f"/api/v1/jobs/{job_id}/logs"
        path = _seed_log(_setup, job_id, [{"level": "INFO", "message": "old"}])
        cursor = client.get(url).headers["x-next-cursor"]
        path.unlink()
        _seed_log(_setup, job_id, [{"level": "INFO", "message": "new"}])
        response = client.get(url, params={"cursor": cursor})
        assert response.headers["x-log-reset"] == "true"
        assert [json.loads(line)["message"] for line in response.text.splitlines()] == ["new"]
        assert client.get(url, params={"cursor": "../../etc/passwd"}).status_code == 422

    def test_invalid_level_returns_422(self, client, _setup):
        _seed_log(
            _setup,
            "job_aaaaaaaa8888",
            [
                {
                    "ts": "2026-04-27T12:00:00+00:00",
                    "level": "INFO",
                    "code": "X",
                    "job_id": "job_aaaaaaaa8888",
                    "message": "x",
                }
            ],
        )
        resp = client.get("/api/v1/jobs/job_aaaaaaaa8888/logs?level=NUCLEAR")
        assert resp.status_code == 422

    def test_invalid_since_returns_422(self, client, _setup):
        _seed_log(
            _setup,
            "job_aaaaaaaa8889",
            [
                {
                    "ts": "2026-04-27T12:00:00+00:00",
                    "level": "INFO",
                    "code": "X",
                    "job_id": "job_aaaaaaaa8889",
                    "message": "x",
                }
            ],
        )
        resp = client.get("/api/v1/jobs/job_aaaaaaaa8889/logs?since=not-a-date")
        assert resp.status_code == 422
