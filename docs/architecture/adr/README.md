# Architecture Decision Records

This directory is reserved for future ADRs filed after v4.1.0. Existing ADRs
live one level up at `docs/architecture/adr-*.md`:

- [`../adr-fasttext-assessment.md`](../adr-fasttext-assessment.md) — FastText language detection assessment
- [`../adr-paddlepaddle-upgrade-path.md`](../adr-paddlepaddle-upgrade-path.md) — PaddlePaddle upgrade path
- [`../adr-sqlite-to-postgresql-migration.md`](../adr-sqlite-to-postgresql-migration.md) — SQLite → PostgreSQL migration

New ADRs should follow the template at the bottom of this file and live in this
directory with filenames like `NNNN-short-title.md`.

---

## ADR Template

```markdown
# ADR NNNN: <Title>

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Superseded by ADR NNNN | Deprecated
**Deciders**: <names or "core maintainers">

## Context

What is the issue we're trying to address? What forces are at play?

## Decision

What did we decide? Use active voice, decisive.

## Consequences

What becomes easier? What becomes harder? What new work does this create?

## Alternatives Considered

- **Option A** — Why rejected.
- **Option B** — Why rejected.

## References

- Related issues, PRs, prior ADRs, external papers.
```
