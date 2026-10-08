#!/usr/bin/env python3
"""Attempt free/local Qwen and COMETKiwi runtime quality probes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cometkiwi", "qwen"], required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--inject-truststore", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _emit(payload: dict, output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def run_cometkiwi(model_id: str) -> dict:
    model_id = model_id or os.environ.get(
        "COMETKIWI_MODEL_ID", "Unbabel/wmt22-cometkiwi-da"
    )
    samples = [
        {"src": "Hello world.", "mt": "Bonjour le monde."},
        {"src": "The file is ready.", "mt": "Le fichier est pret."},
    ]
    try:
        from comet import download_model, load_from_checkpoint

        checkpoint = download_model(model_id)
        model = load_from_checkpoint(checkpoint)
        scores = model.predict(samples, batch_size=2, gpus=0).scores
    except Exception as exc:
        return {
            "mode": "cometkiwi",
            "status": "blocked_runtime",
            "model": model_id,
            "command": "python scripts/run_translation_quality_runtime_probe.py --mode cometkiwi",
            "hardware": "CPU accepted for proof if model loads; GPU recommended for production throughput",
            "failure": repr(exc),
        }
    return {
        "mode": "cometkiwi",
        "status": "passed",
        "model": model_id,
        "scores": list(scores),
    }


def run_qwen(model_id: str) -> dict:
    model_id = model_id or os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    prompt = "Translate to French: Hello world."
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=24)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:
        return {
            "mode": "qwen",
            "status": "blocked_runtime",
            "model": model_id,
            "command": "python scripts/run_translation_quality_runtime_probe.py --mode qwen --model Qwen/Qwen2.5-0.5B-Instruct",
            "hardware": "CPU can prove runtime for 0.5B class model; GPU required for production latency/SLO evidence",
            "failure": repr(exc),
        }
    return {"mode": "qwen", "status": "passed", "model": model_id, "output": decoded}


def _inject_truststore_if_available() -> str:
    try:
        import truststore
    except Exception as exc:  # pragma: no cover - depends on local env
        return f"truststore_unavailable: {exc!r}"
    truststore.inject_into_ssl()
    return "truststore_injected"


def _resolve_runtime_model(
    model_id: str, local_dir: Path | None
) -> tuple[str, str | None]:
    if local_dir is not None and local_dir.exists():
        return str(local_dir), str(local_dir)
    return model_id, None


def run_qwen_runtime(
    model_id: str,
    *,
    local_dir: Path | None,
    inject_truststore: bool,
) -> dict:
    model_id = model_id or os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    truststore_status = (
        _inject_truststore_if_available() if inject_truststore else "not_requested"
    )
    runtime_model, resolved_local_dir = _resolve_runtime_model(model_id, local_dir)
    prompt = "Translate to French: Hello world."
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_only = bool(resolved_local_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            runtime_model, local_files_only=local_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            runtime_model, local_files_only=local_only
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=24)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:
        return {
            "mode": "qwen",
            "status": "blocked_runtime",
            "model": model_id,
            "runtime_model": runtime_model,
            "local_dir": resolved_local_dir,
            "truststore": truststore_status,
            "command": "python scripts/run_translation_quality_runtime_probe.py --mode qwen --model Qwen/Qwen2.5-0.5B-Instruct",
            "hardware": "CPU can prove runtime for 0.5B class model; GPU required for production latency/SLO evidence",
            "failure": repr(exc),
        }
    return {
        "mode": "qwen",
        "status": "passed",
        "model": model_id,
        "runtime_model": runtime_model,
        "local_dir": resolved_local_dir,
        "truststore": truststore_status,
        "hardware": "CPU local/dev runtime proof only; GPU and approved model policy required for production SLO evidence",
        "output": decoded,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = (
        run_cometkiwi(args.model)
        if args.mode == "cometkiwi"
        else run_qwen_runtime(
            args.model,
            local_dir=args.local_dir,
            inject_truststore=args.inject_truststore,
        )
    )
    _emit(payload, args.output)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
