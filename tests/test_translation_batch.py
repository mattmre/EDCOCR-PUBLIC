"""Unit tests for ``ocr_local.translation.batch`` (Plan B Wave M2 -- B17).

These tests run without Django/Celery configured.  Validation tests
exercise the pure-Python invariants on :class:`BatchTranslationRequest`;
persistence tests patch :func:`ocr_local.translation.batch._load_django_models`
with an in-memory fake to keep the unit-test lane runnable in CI without
DJANGO_SETTINGS_MODULE.
"""
from __future__ import annotations

from typing import Iterable
from unittest.mock import MagicMock

import pytest

from ocr_local.translation.batch import (
    DEFAULT_INPUT_MAX_BYTES,
    BatchInput,
    BatchNotFoundError,
    BatchTranslationRequest,
    BatchValidationError,
    cancel_batch,
    collect_results,
    fan_out,
    from_dict,
    get_status,
    submit_batch,
    to_dict,
)
from ocr_local.translation.policy import PolicyDenied

# ---------------------------------------------------------------------------
# In-memory Django-model fake
# ---------------------------------------------------------------------------


class _FakeJobRow:
    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_CANCELLED = "cancelled"

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.submitted_at = None
        self.completed_at = None

    def save(self, update_fields=None):
        return None


class _FakeInputRow:
    def __init__(self, **kw):
        self.batch = kw.get("batch")
        self.client_ref = kw["client_ref"]
        self.input_index = kw["input_index"]
        self.text = kw["text"]
        self.status = kw.get("status", "pending")
        self.target_text = kw.get("target_text", "")
        self.engine_id = kw.get("engine_id", "")
        self.confidence = kw.get("confidence")
        self.glossary_hits_json = kw.get("glossary_hits_json") or []
        self.error = kw.get("error", "")
        self.celery_task_id = kw.get("celery_task_id", "")
        self.started_at = kw.get("started_at")
        self.completed_at = kw.get("completed_at")
        self.optional_metadata_json = kw.get("optional_metadata_json") or {}

    def save(self, update_fields=None):
        return None


class _FakeQuerySet:
    def __init__(self, items: Iterable):
        self._items = list(items)

    def filter(self, **kw):
        out = []
        for item in self._items:
            ok = True
            for k, v in kw.items():
                if k.endswith("__in"):
                    real_k = k[: -len("__in")]
                    if getattr(item, real_k, None) not in v:
                        ok = False
                        break
                elif getattr(item, k, None) != v:
                    ok = False
                    break
            if ok:
                out.append(item)
        return _FakeQuerySet(out)

    def order_by(self, key):
        descending = key.startswith("-")
        if descending:
            key = key[1:]
        return _FakeQuerySet(
            sorted(self._items, key=lambda r: getattr(r, key), reverse=descending),
        )

    def values_list(self, field, flat=False):
        if flat:
            return [getattr(r, field) for r in self._items]
        return [(getattr(r, field),) for r in self._items]

    def count(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class _FakeJobManager:
    TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    class DoesNotExist(Exception):
        pass

    def __init__(self):
        self._rows: list[_FakeJobRow] = []

    def create(self, **kw):
        row = _FakeJobRow(**kw)
        self._rows.append(row)
        return row

    def get(self, batch_id=None, **kw):
        for r in self._rows:
            if r.batch_id == batch_id:
                return r
        raise self.DoesNotExist(batch_id)

    def all(self):
        return _FakeQuerySet(self._rows)


class _FakeInputManager:
    def __init__(self):
        self._rows: list[_FakeInputRow] = []

    def create(self, **kw):
        row = _FakeInputRow(**kw)
        self._rows.append(row)
        return row

    def filter(self, **kw):
        return _FakeQuerySet(self._rows).filter(**kw)


class _FakeJobModel:
    """Stand-in for the BatchTranslationJob Django class."""

    DoesNotExist = _FakeJobManager.DoesNotExist
    TERMINAL_STATUSES = _FakeJobManager.TERMINAL_STATUSES
    STATUS_PENDING = _FakeJobManager.STATUS_PENDING
    STATUS_RUNNING = _FakeJobManager.STATUS_RUNNING
    STATUS_COMPLETED = _FakeJobManager.STATUS_COMPLETED
    STATUS_FAILED = _FakeJobManager.STATUS_FAILED
    STATUS_CANCELLED = _FakeJobManager.STATUS_CANCELLED

    def __init__(self):
        self.objects = _FakeJobManager()


class _FakeInputModel:
    DoesNotExist = type("DoesNotExist", (Exception,), {})

    def __init__(self):
        self.objects = _FakeInputManager()


@pytest.fixture
def fake_models(monkeypatch):
    """Patch ``_load_django_models`` and ``django.db.transaction``."""
    job = _FakeJobModel()
    inp = _FakeInputModel()

    monkeypatch.setattr(
        "ocr_local.translation.batch._load_django_models",
        lambda: (job, inp),
    )

    # Stub django.db.transaction.atomic with a no-op context manager so
    # submit_batch's ``with transaction.atomic():`` works.
    import contextlib
    import sys
    import types

    fake_django = types.ModuleType("django")
    fake_db = types.ModuleType("django.db")
    fake_db.transaction = types.SimpleNamespace(
        atomic=contextlib.nullcontext,
    )
    fake_django.db = fake_db
    sys.modules.setdefault("django", fake_django)
    sys.modules["django.db"] = fake_db
    return job, inp


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _req(**overrides) -> BatchTranslationRequest:
    base = dict(
        tenant_id="tenant-a",
        source_lang="en",
        target_lang="fr",
        inputs=[BatchInput(client_ref="r1", text="hello")],
    )
    base.update(overrides)
    return BatchTranslationRequest(**base)


def test_validate_rejects_empty_tenant():
    r = _req(tenant_id="")
    with pytest.raises(BatchValidationError):
        submit_batch(r)


def test_validate_rejects_missing_source_lang():
    r = _req(source_lang="")
    with pytest.raises(BatchValidationError):
        submit_batch(r)


def test_validate_rejects_missing_target_lang():
    r = _req(target_lang="")
    with pytest.raises(BatchValidationError):
        submit_batch(r)


def test_validate_rejects_empty_inputs():
    r = _req(inputs=[])
    with pytest.raises(BatchValidationError):
        submit_batch(r)


def test_validate_rejects_too_many_inputs(fake_models):
    inputs = [BatchInput(client_ref=f"r{i}", text="x") for i in range(11)]
    r = _req(inputs=inputs)
    with pytest.raises(BatchValidationError):
        submit_batch(r, max_inputs=10)


def test_validate_rejects_duplicate_client_refs(fake_models):
    inputs = [
        BatchInput(client_ref="dup", text="a"),
        BatchInput(client_ref="dup", text="b"),
    ]
    r = _req(inputs=inputs)
    with pytest.raises(BatchValidationError):
        submit_batch(r)


def test_validate_rejects_oversize_input(fake_models):
    big_text = "x" * (DEFAULT_INPUT_MAX_BYTES + 1)
    inputs = [BatchInput(client_ref="r1", text=big_text)]
    r = _req(inputs=inputs)
    with pytest.raises(BatchValidationError):
        submit_batch(r)


def test_validate_rejects_priority_out_of_range(fake_models):
    r = _req(priority=10)
    with pytest.raises(BatchValidationError):
        submit_batch(r)


def test_validate_rejects_negative_priority(fake_models):
    r = _req(priority=-1)
    with pytest.raises(BatchValidationError):
        submit_batch(r)


def test_validate_rejects_missing_client_ref(fake_models):
    inputs = [BatchInput(client_ref="", text="x")]
    r = _req(inputs=inputs)
    with pytest.raises(BatchValidationError):
        submit_batch(r)


# ---------------------------------------------------------------------------
# Certified rejection (gotcha #87 -- emit BEFORE raise)
# ---------------------------------------------------------------------------


def test_certified_true_raises_policy_denied(fake_models):
    r = _req(requested_certified=True)
    with pytest.raises(PolicyDenied) as exc_info:
        submit_batch(r)
    # ReasonCode.BATCH_REJECTED_CERTIFIED comes through.
    assert "BATCH_REJECTED_CERTIFIED" in str(exc_info.value)


def test_certified_true_emits_custody_event_before_raise(fake_models):
    """The custody event MUST be logged BEFORE the raise (gotcha #87)."""
    chain = MagicMock()
    r = _req(requested_certified=True)
    with pytest.raises(PolicyDenied):
        submit_batch(r, custody_chain=chain)
    # log_event was called exactly once with BATCH_REJECTED_CERTIFIED.
    assert chain.log_event.call_count == 1
    args, _ = chain.log_event.call_args
    assert args[0] == "BATCH_REJECTED_CERTIFIED"
    payload = args[1]
    assert payload["tenant_id"] == "tenant-a"
    assert payload["source_lang"] == "en"
    assert payload["target_lang"] == "fr"


def test_certified_true_does_not_persist(fake_models):
    job, _inp = fake_models
    r = _req(requested_certified=True)
    with pytest.raises(PolicyDenied):
        submit_batch(r)
    assert len(job.objects._rows) == 0


# ---------------------------------------------------------------------------
# submit_batch happy path
# ---------------------------------------------------------------------------


def test_submit_batch_returns_uuid_hex(fake_models):
    bid = submit_batch(_req())
    assert isinstance(bid, str)
    # uuid4().hex -> 32 hex chars.
    assert len(bid) == 32
    int(bid, 16)  # must parse as hex


def test_submit_batch_persists_job_row(fake_models):
    job, _inp = fake_models
    bid = submit_batch(_req())
    assert len(job.objects._rows) == 1
    row = job.objects._rows[0]
    assert row.batch_id == bid
    assert row.tenant_id == "tenant-a"
    assert row.source_lang == "en"
    assert row.target_lang == "fr"
    assert row.status == "pending"
    assert row.total_inputs == 1
    assert row.completed_inputs == 0
    assert row.failed_inputs == 0


def test_submit_batch_persists_input_rows(fake_models):
    _job, inp = fake_models
    bid = submit_batch(
        _req(inputs=[
            BatchInput(client_ref="a", text="alpha"),
            BatchInput(client_ref="b", text="beta"),
        ])
    )
    rows = inp.objects._rows
    assert len(rows) == 2
    assert rows[0].client_ref == "a"
    assert rows[0].input_index == 0
    assert rows[0].text == "alpha"
    assert rows[1].client_ref == "b"
    assert rows[1].input_index == 1
    # Both linked to the same job row.
    assert rows[0].batch.batch_id == bid
    assert rows[1].batch.batch_id == bid


def test_submit_batch_emits_custody_submitted(fake_models):
    chain = MagicMock()
    bid = submit_batch(_req(), custody_chain=chain)
    assert chain.log_event.call_count == 1
    args, _ = chain.log_event.call_args
    assert args[0] == "BATCH_SUBMITTED"
    payload = args[1]
    assert payload["batch_id"] == bid
    assert payload["tenant_id"] == "tenant-a"
    assert payload["input_count"] == 1


# ---------------------------------------------------------------------------
# fan_out
# ---------------------------------------------------------------------------


def test_fan_out_unknown_batch_raises(fake_models):
    with pytest.raises(BatchNotFoundError):
        fan_out("nonexistent")


def test_fan_out_dispatches_pending_inputs(fake_models, monkeypatch):
    bid = submit_batch(_req(inputs=[
        BatchInput(client_ref="a", text="alpha"),
        BatchInput(client_ref="b", text="beta"),
    ]))

    captured: list[dict] = []

    class _AsyncResult:
        def __init__(self, idx):
            self.id = f"task-{idx}"

    counter = {"n": 0}

    def fake_apply_async(kwargs=None, queue=None, priority=None, **kw):
        counter["n"] += 1
        captured.append(
            {"kwargs": kwargs, "queue": queue, "priority": priority},
        )
        return _AsyncResult(counter["n"])

    fake_task = MagicMock()
    fake_task.apply_async = fake_apply_async
    fake_module = MagicMock()
    fake_module.translate_batch_input = fake_task

    import sys
    sys.modules["coordinator.jobs.tasks_translation_batch"] = fake_module
    # Also register the parent so import-resolution works.
    sys.modules.setdefault("coordinator", MagicMock())
    sys.modules.setdefault("coordinator.jobs", MagicMock())

    dispatched = fan_out(bid)
    assert dispatched == 2
    assert len(captured) == 2
    assert all(c["queue"] == "translation_batch" for c in captured)
    assert {c["kwargs"]["client_ref"] for c in captured} == {"a", "b"}
    assert all(c["kwargs"]["batch_id"] == bid for c in captured)
    assert all(c["kwargs"]["source_lang"] == "en" for c in captured)
    assert all(c["kwargs"]["target_lang"] == "fr" for c in captured)


def test_fan_out_idempotent_on_terminal_batch(fake_models):
    job, _inp = fake_models
    bid = submit_batch(_req())
    job.objects._rows[0].status = "completed"
    assert fan_out(bid) == 0


def test_fan_out_emits_custody(fake_models):
    bid = submit_batch(_req())
    chain = MagicMock()
    # Pre-set translate_batch_input lookup to a stub that returns a fake task id.
    import sys
    fake_task = MagicMock()
    fake_task.apply_async.return_value = MagicMock(id="task-1")
    fake_module = MagicMock()
    fake_module.translate_batch_input = fake_task
    sys.modules["coordinator.jobs.tasks_translation_batch"] = fake_module

    fan_out(bid, custody_chain=chain)
    fan_out_calls = [
        c for c in chain.log_event.call_args_list if c.args[0] == "BATCH_FAN_OUT"
    ]
    assert len(fan_out_calls) == 1
    payload = fan_out_calls[0].args[1]
    assert payload["batch_id"] == bid
    assert payload["dispatched_count"] >= 1


# ---------------------------------------------------------------------------
# get_status / collect_results
# ---------------------------------------------------------------------------


def test_get_status_unknown_batch_raises(fake_models):
    with pytest.raises(BatchNotFoundError):
        get_status("nope")


def test_get_status_returns_counts(fake_models):
    _job, inp = fake_models
    bid = submit_batch(_req(inputs=[
        BatchInput(client_ref="a", text="alpha"),
        BatchInput(client_ref="b", text="beta"),
        BatchInput(client_ref="c", text="gamma"),
    ]))
    inp.objects._rows[0].status = "completed"
    inp.objects._rows[1].status = "failed"

    snap = get_status(bid)
    assert snap.batch_id == bid
    assert snap.total_inputs == 3
    assert snap.completed_inputs == 1
    assert snap.failed_inputs == 1
    assert snap.pending_inputs == 1
    assert snap.running_inputs == 0


def test_collect_results_unknown_batch_raises(fake_models):
    with pytest.raises(BatchNotFoundError):
        collect_results("nope")


def test_collect_results_in_input_order(fake_models):
    _job, inp = fake_models
    bid = submit_batch(_req(inputs=[
        BatchInput(client_ref="r1", text="t1"),
        BatchInput(client_ref="r2", text="t2"),
    ]))
    inp.objects._rows[0].target_text = "TXT-1"
    inp.objects._rows[0].engine_id = "engine-x"
    inp.objects._rows[1].target_text = "TXT-2"
    inp.objects._rows[1].engine_id = "engine-x"

    results = collect_results(bid)
    assert len(results) == 2
    assert results[0].client_ref == "r1"
    assert results[0].target_text == "TXT-1"
    assert results[1].client_ref == "r2"
    assert results[1].target_text == "TXT-2"


# ---------------------------------------------------------------------------
# cancel_batch
# ---------------------------------------------------------------------------


def test_cancel_batch_unknown_raises(fake_models):
    with pytest.raises(BatchNotFoundError):
        cancel_batch("nope")


def test_cancel_batch_terminal_returns_zero(fake_models):
    job, _inp = fake_models
    bid = submit_batch(_req())
    job.objects._rows[0].status = "completed"
    assert cancel_batch(bid) == 0


def test_cancel_batch_marks_inputs_and_job(fake_models, monkeypatch):
    job, inp = fake_models
    bid = submit_batch(_req(inputs=[
        BatchInput(client_ref="a", text="alpha"),
        BatchInput(client_ref="b", text="beta"),
    ]))
    inp.objects._rows[0].celery_task_id = "task-a"
    inp.objects._rows[0].status = "running"
    inp.objects._rows[1].celery_task_id = "task-b"
    inp.objects._rows[1].status = "pending"

    # Stub coordinator.coordinator.celery to avoid pulling in real broker.
    import sys
    import types

    fake_app = types.SimpleNamespace(
        control=types.SimpleNamespace(revoke=MagicMock()),
    )
    fake_celery_mod = types.ModuleType(
        "coordinator.coordinator.celery",
    )
    fake_celery_mod.app = fake_app
    sys.modules["coordinator.coordinator.celery"] = fake_celery_mod
    sys.modules.setdefault("coordinator", types.ModuleType("coordinator"))
    sys.modules.setdefault(
        "coordinator.coordinator", types.ModuleType("coordinator.coordinator"),
    )

    revoked = cancel_batch(bid)
    assert revoked == 2
    # Both rows now ``cancelled``.
    assert inp.objects._rows[0].status == "cancelled"
    assert inp.objects._rows[1].status == "cancelled"
    assert job.objects._rows[0].status == "cancelled"
    # revoke called for both task ids.
    assert fake_app.control.revoke.call_count == 2


def test_cancel_batch_emits_custody(fake_models):
    bid = submit_batch(_req())
    chain = MagicMock()
    cancel_batch(bid, custody_chain=chain)
    cancel_calls = [
        c for c in chain.log_event.call_args_list if c.args[0] == "BATCH_CANCELLED"
    ]
    assert len(cancel_calls) == 1
    payload = cancel_calls[0].args[1]
    assert payload["batch_id"] == bid


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_to_dict_round_trip():
    r = _req(
        inputs=[
            BatchInput(client_ref="a", text="alpha", optional_metadata={"k": 1}),
            BatchInput(client_ref="b", text="beta"),
        ],
        priority=3,
        glossary_enabled=False,
    )
    d = to_dict(r)
    r2 = from_dict(d)
    assert r2.tenant_id == r.tenant_id
    assert r2.source_lang == r.source_lang
    assert r2.target_lang == r.target_lang
    assert len(r2.inputs) == 2
    assert r2.inputs[0].client_ref == "a"
    assert r2.inputs[0].text == "alpha"
    assert r2.inputs[0].optional_metadata == {"k": 1}
    assert r2.priority == 3
    assert r2.glossary_enabled is False
