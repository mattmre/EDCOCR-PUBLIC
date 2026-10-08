# Local development

Use Python 3.12 for runtime development and Node 22 for the frontend and
TypeScript SDK. Install dependencies in an isolated virtual environment.
Optional OCR assets and native libraries are best exercised in the CPU Docker
image. Runtime evidence must identify the source fingerprint and model subset.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r coordinator/requirements.txt
pip install pytest pytest-asyncio pytest-django pytest-timeout ruff
python scripts/check_version_consistency.py
ruff check api ocr_local ocr_distributed coordinator sdk/python/src
pytest tests/ -q
(cd coordinator && pytest -q)
(cd frontend && npm ci && npm test && npm run build)
(cd sdk/typescript && npm ci && npm test && npm run build:check && npm run build)
pip install build
python -m build sdk/python
```

Some tests need optional models, fonts, Docker Compose or a configured target.
Record skips and failed environment checks explicitly; they do not certify those
surfaces. A fresh real-document API run, restart/recovery, artifact validation,
and package installation remain separate acceptance checks.

Build CPU images through `scripts/with_runtime_source.sh root-cpu` and
coordinator images through its `coordinator` profile. See [INSTALL.md](INSTALL.md)
for the supported candidate flow. Keep one writer per standalone database and
preserve source evidence on every error path.

Hosted workflows are distribution automation and convenience checks. Local
command outputs, artifacts and independent review establish release acceptance.
