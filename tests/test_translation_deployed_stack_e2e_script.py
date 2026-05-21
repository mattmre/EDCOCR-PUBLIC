from __future__ import annotations

import requests

from scripts import run_translation_deployed_stack_e2e as deployed


def test_deployed_stack_env_sets_fail_closed_translation_flags(tmp_path):
    args = deployed.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--api-port",
            "8111",
            "--frontend-port",
            "3111",
            "--postgres-port",
            "55433",
            "--rabbit-port",
            "5679",
            "--redis-port",
            "56380",
        ]
    )
    env = deployed.env_for_stack(args)
    operator_placeholder = "placeholder"
    operator_role = "operator"
    assert env["ENABLE_TRANSLATION_API"] == "true"
    assert env["DJANGO_SETTINGS_MODULE"] == "coordinator.settings"
    assert env["OCR_API_KEY"] == operator_placeholder
    assert env["APIKEY_ROLE"] == operator_role
    assert env["TRANSLATION_MODEL_CACHE_DIR"]
    assert "8111" not in env["DATABASE_URL"]
    assert ":5679//" in env["CELERY_BROKER_URL"]


def test_endpoint_matrix_marks_certify_when_404():
    calls = [
        {"path": "/api/v1/review/rev_abc123abc123/certify", "status_code": 404},
        {"path": "/api/v1/translation/tenants/t/config", "status_code": 200},
    ]
    matrix = deployed.endpoint_matrix(calls)
    assert "review certify" in matrix
    assert "404" in matrix


def test_sha256_manifest_uses_relative_paths(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"abc")
    manifest = deployed.sha256_manifest(model_dir)
    assert set(manifest) == {"model.bin"}
    assert manifest["model.bin"] == deployed.hashlib.sha256(b"abc").hexdigest()


def test_seed_job_id_matches_api_contract():
    source = deployed.seed_api_records.__code__.co_consts
    assert "job_e2e2abc12345" in source


def test_call_api_records_request_exception(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("slow endpoint")

    monkeypatch.setattr(deployed.requests, "request", fail)
    result = deployed.call_api("http://127.0.0.1:1", "GET", "/x")
    assert result["status_code"] == 0
    assert result["response"]["error"] == "Timeout"


def test_call_api_sends_operator_key(monkeypatch):
    observed = {}

    class Response:
        status_code = 200

        def json(self):
            return {"ok": True}

    def fake_request(method, url, **kwargs):
        observed["headers"] = kwargs.get("headers")
        return Response()

    monkeypatch.setattr(deployed.requests, "request", fake_request)
    operator_placeholder = "placeholder"
    result = deployed.call_api(
        "http://127.0.0.1:1", "GET", "/x", api_key=operator_placeholder
    )
    assert result["status_code"] == 200
    assert observed["headers"] == {"X-API-Key": operator_placeholder}


def test_call_api_redacts_sensitive_request_fields(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(deployed.requests, "request", lambda *args, **kwargs: Response())
    operator_placeholder = "placeholder"
    result = deployed.call_api(
        "http://127.0.0.1:1",
        "POST",
        "/x",
        json_body={
            "auth_token": operator_placeholder,
            "nested": [{"password": operator_placeholder}],
        },
        api_key=operator_placeholder,
    )

    assert result["request"] == {
        "auth_token": "<redacted>",
        "nested": [{"password": "<redacted>"}],
    }


def test_find_free_port_skips_open_port(monkeypatch):
    monkeypatch.setattr(deployed, "port_open", lambda host, port: port == 5673)
    assert deployed.find_free_port(5673) == 5674


def test_standalone_rabbit_argument_exists():
    args = deployed.parse_args(["--standalone-rabbit", "--rabbit-port", "5688"])
    assert args.standalone_rabbit is True
    assert args.rabbit_port == 5688


def test_resolve_executable_returns_name_for_missing_tool(monkeypatch):
    monkeypatch.setattr(deployed.shutil, "which", lambda name: None)
    assert deployed.resolve_executable("definitely-missing-tool") == "definitely-missing-tool"
