# Contributing to EDCOCR

Thank you for considering a contribution. EDCOCR is an open platform; the codebase improves every time someone outside the original team picks up a piece and pushes it forward.

This document covers how to file issues, how to propose changes, and what we expect from contributions before they merge.

## Table of Contents

1. [Code of Conduct](#1-code-of-conduct)
2. [How to Report Issues](#2-how-to-report-issues)
3. [How to Propose Changes](#3-how-to-propose-changes)
4. [Development Setup](#4-development-setup)
5. [Coding Conventions](#5-coding-conventions)
6. [Testing Expectations](#6-testing-expectations)
7. [Pull Request Process](#7-pull-request-process)
8. [Documentation Expectations](#8-documentation-expectations)
9. [Security Disclosures](#9-security-disclosures)
10. [Joining the Team](#10-joining-the-team)

---

## 1. Code of Conduct

Be civil, be technical, and respect other contributors' time. Disagreements about design are welcome and expected; personal attacks and harassment are not.

If you experience or witness unacceptable behavior, contact the maintainers privately via the security disclosure channel (see [`SECURITY.md`](SECURITY.md)).

---

## 2. How to Report Issues

Before filing a new issue:

- **Search existing issues** for duplicates.
- **Read [`docs/known-issues.md`](docs/known-issues.md)** for documented limitations.
- **Try the latest release** — your bug may already be fixed.

When filing, include:

- EDCOCR version (`cat version.py` or `docker exec <container> python -c "import version; print(version.__version__)"`)
- Deployment method (Docker, Kubernetes, bare metal)
- Python version
- GPU model and driver version (if GPU-related)
- A minimal reproduction (sample PDF if possible — sanitize first!)
- Expected vs actual behavior
- Relevant log excerpts (with secrets redacted)

For feature requests, describe **the problem you want to solve**, not the implementation you want. Solutions get easier when the problem is sharp.

---

## 3. How to Propose Changes

Small fixes (typos, single-file refactors, documentation patches): open a pull request directly.

Larger changes (new features, architectural shifts, dependency upgrades): open an issue first to discuss approach. This saves you from writing 500 lines of code that gets rejected because we wanted a different design.

For changes that touch:

- **`ocr_gpu_async.py`** — the production pipeline. Coordinate with maintainers before starting.
- **`ocr_local/translation/*`** — translation suite is feature-flagged and has specific compliance requirements.
- **`helm/ocr-local/*`** — Helm chart changes require Helm lint + cluster validation.
- **`coordinator/coordinator/settings.py`** — coordinator settings affect every deployment.

These are conflict-prone areas; coordinate to avoid wasted work.

---

## 4. Development Setup

```bash
git clone https://github.com/mattmre/EDCOCR-PUBLIC.git
cd EDCOCR-PUBLIC
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if present
pip install pre-commit
pre-commit install
```

Run the test suite locally before pushing:

```bash
python -m pytest tests/ -v
```

Coordinator tests live separately:

```bash
cd coordinator
python -m pytest tests/ -v
```

For a quick smoke run:

```bash
python scripts/smoke_pipeline.py
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the full development guide.

---

## 5. Coding Conventions

### Python

- **Python 3.10+** language features are fine; we don't support 3.9 anymore.
- **Type hints** on public functions and class methods. Internal helpers can skip them.
- **`from __future__ import annotations`** at the top of new modules.
- **`ruff`** for linting. Run `ruff check .` before committing — pre-commit will catch this.
- **`stdlib logging`** with a module-level logger. Do not use `print()` in production code.
- **Constants at module top** in ALL_CAPS.
- **No silent failures.** Either handle an exception meaningfully or let it propagate.

### TypeScript

- **TypeScript 5.0+** strict mode.
- **No `any`** unless the boundary genuinely is untyped (and even then, prefer `unknown`).
- **ESM modules** (no CommonJS in new code).
- **`prettier`** for formatting.

### Documentation

- **Markdown** for prose. Diagrams use **Mermaid** (which renders on GitHub).
- **Plain English.** Avoid jargon when a plain word works.
- **No emojis** in code, comments, or commit messages unless the team explicitly opts in.

### Commit Messages

Conventional Commits format is preferred:

```
type(scope): short summary

Longer explanation if needed. Wrap at 72 columns.

Refs #123
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`.

Do **not** include AI/LLM co-author footers in commits. If you used an AI assistant, that's fine; the commit is still your work product and your responsibility.

---

## 6. Testing Expectations

Every PR is expected to:

- **Add tests for new code paths.** Unit tests for pure functions, integration tests for orchestration.
- **Update tests for changed code paths.** Don't leave green tests that no longer reflect reality.
- **Keep CI green.** PRs with failing CI will not be merged.
- **Never silence failing tests** by skipping them. Either fix the test or fix the code.

Run before pushing:

```bash
ruff check .
python -m pytest tests/ -v
# For coordinator changes:
cd coordinator && python -m pytest tests/ -v
```

For changes touching the pipeline, also run:

```bash
python scripts/smoke_pipeline.py
```

### Test Patterns

- **Pure functions** → unit tests with explicit input/output assertions
- **Pipeline stages** → integration tests against a real `tests/fixtures/sample.pdf`
- **API endpoints** → FastAPI `TestClient` with auth + rate limit + size limit coverage
- **Worker tasks** → mock Celery broker, assert task signature and result
- **Schema changes** → JSON Schema validation tests
- **Bugfixes** → at least one regression test reproducing the bug before the fix

---

## 7. Pull Request Process

### Before Opening

- Rebase on `main`. PRs based on stale branches are confusing to review.
- Run the full test suite locally.
- Run `ruff check .` and address any new warnings.
- Update relevant docs in `docs/`.
- If you changed a feature flag default, **say so prominently in the PR body**.

### PR Body

Include:

- **What changed and why.** Not "fixed bug" — "fixed off-by-one in page resume gap detection that caused last page to be reprocessed."
- **How it was tested.** Test files, manual steps, CI evidence.
- **Risks.** What could break? Who is affected?
- **Breaking changes.** Flag explicitly. If you changed the API, the env var contract, or the on-disk output layout, the next major version bump may be required.

### Review

- Address **all** review comments. If you disagree, say so and explain — don't just dismiss.
- Push **new commits** to the PR branch. Don't force-push during review (it makes review diff history hard to read). Squash before merge is fine.
- CI must pass on the latest commit.
- At least one maintainer approval is required.

### Merge

Maintainers will squash-merge most PRs. The PR title becomes the squash commit title; clean it up before merge.

---

## 8. Documentation Expectations

If your PR:

- **Adds or removes a feature flag** → update [`docs/06-CONFIGURATION-REFERENCE.md`](docs/06-CONFIGURATION-REFERENCE.md).
- **Adds or changes an API endpoint** → update [`docs/API-REFERENCE.md`](docs/API-REFERENCE.md).
- **Changes the SDK surface** → update [`docs/08-SDK-REFERENCE.md`](docs/08-SDK-REFERENCE.md) and bump SDK package versions.
- **Changes deployment topology** → update [`ARCHITECTURE.md`](ARCHITECTURE.md) and the relevant deployment guide.
- **Changes runtime behavior** → update [`CHANGELOG.md`](CHANGELOG.md).
- **Adds a new dependency** → justify in PR body; we minimize new transitive dependencies.

Docs PRs with no code changes are welcome and don't require maintainer pre-approval.

---

## 9. Security Disclosures

**Do not file public issues for security vulnerabilities.**

Use [GitHub's private vulnerability reporting](https://github.com/mattmre/EDCOCR-PUBLIC/security/advisories/new) to submit a report. The maintainers will respond within five business days.

See [`SECURITY.md`](SECURITY.md) for the full disclosure policy.

---

## 10. Joining the Team

EDCOCR is built as a community-driven project. Drive-by patches, bug reports, and Discussions posts are all valuable — but if you want to do more than that, we want to hear from you.

**If you want to join the team as a regular contributor**, send a direct message to [**@mattmre**](https://github.com/mattmre) on GitHub. Tell us:

- What you want to work on (a specific area, a roadmap item, or "wherever I'm useful").
- A few links to prior work — open source contributions, your own projects, a blog post, anything that shows how you think.
- How much time per week you realistically have. We respect "a few hours on weekends" as much as full-time.

There is no application form, no resume screen, and no NDA. We pair new contributors with an existing maintainer on a small first issue and grow scope from there.

Areas where extra hands are most useful right now:

- **Pipeline performance** — ONNX/OpenVINO tuning, batch sizing, GPU memory profiling.
- **Translation suite** — adapter coverage, glossary tooling, quality estimators.
- **Document Intelligence** — layout analysis, form extraction, table accuracy.
- **Operations / SRE** — Helm chart improvements, observability, multi-cluster topologies.
- **Docs and examples** — the fastest way to help someone the next time they hit the same wall.

If you would rather contribute publicly first and see how it goes, that's also fine — open a PR or a Discussion and we'll meet you there.

---

## Recognition

Contributors are credited in:

- The Git commit log (your authored commits)
- Release notes for releases that include your contributions
- A `CONTRIBUTORS.md` file (in flight — opened on first external merge)

Thank you for making EDCOCR better.
