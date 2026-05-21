# ADR: SQLite WAL Mode and PostgreSQL Migration Path

**Status**: Accepted (WAL mode applied); PostgreSQL migration deferred to Q2 2026
**Date**: 2026-04-07
**Finding IDs**: (WAL mode), (database consolidation)

## Context

The API layer uses three separate SQLite databases:

1. **`api/database.py`** -- Main jobs database (via SQLAlchemy engine, `DB_PATH`)
2. **`api/review_queue.py`** -- Human review queue (raw `sqlite3`, `DB_PATH`)
3. **`api/entity_index.py`** -- Entity/extraction recall index (raw `sqlite3`, `OUTPUT_FOLDER/entity_index.db`)

Each database has its own connection pool, lock management, and file on disk. Under concurrent API load, SQLite's default rollback journal mode blocks all readers while a write is in progress. This can cause `SQLITE_BUSY` errors under moderate concurrency.

## Decision

### Immediate: WAL journal mode on all three databases

Enable WAL (Write-Ahead Logging) mode on every new SQLite connection across all three databases. WAL allows concurrent readers while a single writer operates, which eliminates the most common contention pattern.

Applied PRAGMAs on every connection:

```sql
PRAGMA journal_mode=WAL;           -- Concurrent reads during writes
PRAGMA synchronous=NORMAL;         -- Safe with WAL; avoids fsync on every commit
PRAGMA wal_autocheckpoint=1000;    -- Checkpoint every 1000 pages (~4 MB)
PRAGMA busy_timeout=5000;          -- Wait 5s before returning SQLITE_BUSY
```

**Why `synchronous=NORMAL`**: In WAL mode, `NORMAL` provides durability guarantees equivalent to `FULL` in rollback-journal mode. A crash could lose only the most recent transaction (which would be retried anyway). The performance gain is significant: no fsync on every commit.

### Deferred: Migrate to PostgreSQL

Three separate SQLite files means:
- Three separate connection pools to manage
- Three separate lock management concerns
- Three files for operators to back up
- No shared transaction guarantees across databases
- No connection pooling efficiency

**Migration approach**:
1. Define Alembic migration scripts from current SQLite schemas
2. Consolidate all three schemas into a single PostgreSQL database
3. Use SQLAlchemy for all three (review_queue and entity_index currently use raw `sqlite3`)
4. Connection pool configuration via environment variables
5. Backward-compatible SQLite fallback for development/testing

**Timeline**: Q2 2026, after the P2 remediation sprint closes. The coordinator already uses PostgreSQL via Django ORM, so the infrastructure patterns exist.

## Consequences

### Positive
- WAL mode immediately reduces `SQLITE_BUSY` errors under concurrent API load
- `busy_timeout=5000` provides automatic retry instead of immediate failure
- `synchronous=NORMAL` improves write throughput without sacrificing WAL-mode durability
- ADR documents the clear path to PostgreSQL when scaling demands it

### Negative
- WAL mode creates `-wal` and `-shm` sidecar files alongside each `.db` file
- Operators must include `-wal` and `-shm` files in backups (or checkpoint before backup)
- Three separate databases remain until the PostgreSQL migration

### Risks
- WAL mode is not compatible with network file systems (NFS) -- the API SQLite databases must remain on local storage
- WAL mode slightly increases disk usage (WAL file can grow before checkpoint)

## References

- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
- [SQLite PRAGMA synchronous](https://www.sqlite.org/pragma.html#pragma_synchronous)
- Expert panel finding (WAL mode)
- Expert panel finding (database consolidation)
