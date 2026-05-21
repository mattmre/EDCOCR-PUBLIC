"""Tests for local translation quality runtime probe wiring."""

from __future__ import annotations

import types

from scripts import run_translation_quality_runtime_probe as probe


def test_qwen_runtime_uses_local_dir_and_offline(monkeypatch, tmp_path):
    calls = []

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, model, *, local_files_only=False):
            calls.append(("tokenizer", model, local_files_only))
            return cls()

        def __call__(self, prompt, return_tensors=None):
            return {"input_ids": [1, 2, 3]}

        def decode(self, tokens, skip_special_tokens=True):
            return "Translate to French: Hello world. Bonjour le monde."

    class _Model:
        @classmethod
        def from_pretrained(cls, model, *, local_files_only=False):
            calls.append(("model", model, local_files_only))
            return cls()

        def generate(self, **kwargs):
            return [[1, 2, 3, 4]]

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_Model,
        AutoTokenizer=_Tokenizer,
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    local_dir = tmp_path / "qwen"
    local_dir.mkdir()
    result = probe.run_qwen_runtime(
        "tiny-random/qwen2.5",
        local_dir=local_dir,
        inject_truststore=False,
    )

    assert result["status"] == "passed"
    assert result["runtime_model"] == str(local_dir)
    assert result["local_dir"] == str(local_dir)
    assert result["truststore"] == "not_requested"
    assert ("tokenizer", str(local_dir), True) in calls
    assert ("model", str(local_dir), True) in calls
