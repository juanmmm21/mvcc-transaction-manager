# mvcc-transaction-manager

**Component 4 of 11 (Transactions & Concurrency, 1 of 2)** in the [`strata-database-engine`](https://github.com/juanmmm21/strata-database-engine) ecosystem.
Repo: [`github.com/juanmmm21/mvcc-transaction-manager`](https://github.com/juanmmm21/mvcc-transaction-manager)

A from-scratch multiversion concurrency control (MVCC) manager, written in Python. It keeps consistent snapshots per transaction, implements the three classic SQL isolation levels (read committed, repeatable read, serializable), and never lets a reader block a concurrent writer — the defining trade-off MVCC makes against two-phase locking.

---

## What it is and what problem it solves

Two transactions touching the same rows can interact in ways that quietly corrupt an application's invariants: one reads data another hasn't committed yet (dirty read), one gets a different answer to the same query twice (non-repeatable read), a row appears out of nowhere on a re-scan (phantom read), or two transactions each make a locally-valid decision that is jointly wrong (write skew). SQL's isolation levels are a contract about which of these a database will prevent.

This project implements that contract from scratch, without table-level or row-level locking on the read path: every transaction gets its own consistent view of the data (a *snapshot*), built by comparing a monotonically increasing commit sequence number against a horizon frozen at the right moment for each isolation level. Writes are never blocked by readers; conflicting writers are resolved optimistically, at commit time, by aborting the loser instead of making anyone wait.

## Role in `strata-database-engine`

```text
write-ahead-log-recovery ──┬─▶ bplus-tree-storage-engine ─┐
                            └─▶ lsm-tree-engine ────────────┤
                                                             ▼
                                          ┌───────────────────────────────┐
                                          │   mvcc-transaction-manager    │  (this repo)
                                          │  snapshots · isolation levels  │
                                          │  write-skew detection · GC     │
                                          └───────────────┬────────────────┘
                                                           │ implements RowStore
                                                           ▼
                                                        nanosql
```

This repo does **not** import `bplus-tree-storage-engine` or `lsm-tree-engine` — the real integration happens only inside `nanosql`. Instead it defines its own minimal `RowStore` `Protocol` (`get`/`put`-style access to versioned rows) and ships an in-memory implementation, `InMemoryRowStore`, good enough to exercise every concurrency-control rule in isolation. Any adapter over a real storage engine that satisfies `RowStore` can replace it without touching `pipeline.py`. It is a sibling, alternative concurrency-control strategy to [`lock-manager-deadlock-detector`](https://github.com/juanmmm21/lock-manager-deadlock-detector) — `nanosql` picks one or the other per configuration.

## Goal / skills demonstrated

- Snapshot-based visibility without locking readers against writers, the mechanism behind Postgres, MySQL InnoDB and CockroachDB.
- The three standard SQL isolation levels implemented as three different rules over the same version chain, not three different code paths.
- First-committer-wins write-write conflict detection.
- A simplified but real implementation of **Serializable Snapshot Isolation** (Cahill, Röhm & Fekete, 2008): detecting write skew by tracking read/write antidependencies and aborting the transaction that would close a serialization cycle.
- Garbage collection of row versions bounded by the oldest snapshot still in use — a live, always-correct alternative to a periodic full vacuum.
- Concurrency correctness demonstrated with real OS threads, not just simulated single-threaded interleaving.

## How it works

### Versioning and visibility

Every row version carries only two facts: `created_by` (the id of the transaction that wrote it) and `commit_seq`, a counter incremented once per successful commit. A transaction's snapshot is a single number, `commit_horizon`: any version with `commit_seq <= commit_horizon` is visible to it.

This is simpler than Postgres's `xmin`/`xmax`-plus-active-list scheme because of one invariant this module enforces strictly: **a transaction's writes are buffered locally and only appended to the shared store inside the critical section of a successful `commit`.** No other transaction ever observes a half-committed version, so visibility never needs to consult a commit-status table — it is a single integer comparison.

| Isolation level | Snapshot | Effect |
|---|---|---|
| `READ_COMMITTED` | recomputed on every read | dirty read: never. Non-repeatable / phantom read: permitted (each read sees the latest committed state). |
| `REPEATABLE_READ` | fixed at `begin`, reused for the whole transaction | dirty, non-repeatable and phantom read: all prevented (the snapshot never advances — Postgres-style, stronger than the SQL standard's minimum for this level). |
| `SERIALIZABLE` | same fixed snapshot as `REPEATABLE_READ` | everything above, plus write-skew detection (see below). |

### Write-write conflicts (first-committer-wins)

In `REPEATABLE_READ` and `SERIALIZABLE`, a transaction that tries to commit a write to a row whose latest committed version has a `commit_seq` newer than its own snapshot horizon aborts with `WriteConflictError`: someone else already changed data it never saw. `READ_COMMITTED` skips this check — every one of its reads already targets the latest committed state, so there is no "stale" write to protect against.

### Write skew (Serializable Snapshot Isolation, simplified)

Write-write conflict detection alone does not stop write skew: two transactions can read the same rows and write to two *different* rows, each individually valid, jointly wrong (the classic example: two on-call doctors, each signs off only if the other is still on call — both read `on_call=true` for the other, both sign off, nobody is left on call).

`SERIALIZABLE` tracks, per transaction, the set of rows read and written. At commit time it checks for two antidependencies:

- **Incoming:** some other transaction, concurrent with this one, already committed a write to a row this transaction read.
- **Outgoing:** this transaction is about to write a row that some other concurrent transaction (still active, or already committed) had read.

A transaction with **both** an incoming and an outgoing antidependency is a *pivot*: it sits in the middle of a potential serialization cycle and aborts with `SerializationConflictError`, which is enough to break any cycle that would pass through it — without building the full dependency graph. In the two-doctor example, whichever transaction commits second finds both edges against the first and aborts.

### Garbage collection

`gc()` computes the horizon as the minimum snapshot among currently active transactions (or "infinity" if none are active) and, per row, keeps only the newest version at or before that horizon plus everything newer — discarding what no active or future transaction could ever need. A row whose newest surviving version is a tombstone and has nothing newer disappears entirely. The same horizon retires the conflict-detection bookkeeping (`read_set`/`write_set`) of committed transactions old enough that no active transaction could still consider them concurrent.

## Architecture

```text
src/mvcc_transaction_manager/
├── __init__.py     # re-exported public API
├── models.py       # RowId, TransactionId, CommitSeq, IsolationLevel, RowVersion, Snapshot,
│                    # Transaction, exception hierarchy
├── protocols.py     # RowStore: the pluggable underlying storage contract
├── store.py          # InMemoryRowStore: reference implementation used by tests and the CLI
├── pipeline.py         # MVCCTransactionManager: snapshots, conflict detection, GC
└── __main__.py         # demonstration CLI (demo / repl / benchmark)
```

- **`models.py`** — immutable data types (`dataclass(frozen=True, slots=True)`) for versions and snapshots, a mutable `Transaction` for in-flight state, and the exception hierarchy (`MvccError` → `TransactionNotActiveError` / `WriteConflictError` / `SerializationConflictError`).
- **`protocols.py`** — `RowStore`, the single coupling point a real storage engine adapter would implement.
- **`store.py`** — `InMemoryRowStore`, a plain `dict`-backed implementation with no internal locking of its own.
- **`pipeline.py`** — all the concurrency-control logic: `begin`/`get`/`put`/`delete`/`scan`/`commit`/`abort`/`gc`.
- **`__main__.py`** — CLI with a `demo` (scripted anomaly walkthrough), an interactive multi-transaction `repl`, and a contention `benchmark`.

**Concurrency:** `MVCCTransactionManager` guards all shared state — the transaction table and the two monotonic counters — with a single `threading.Lock` wrapping every public method. There is no per-row locking: this module's concurrency control is purely optimistic (MVCC), never pessimistic. Blocking, lock-based serialization of individual rows is the job of the sibling subproject, [`lock-manager-deadlock-detector`](https://github.com/juanmmm21/lock-manager-deadlock-detector), not this one.

## Requirements and installation

- Python `>=3.11`

```bash
git clone https://github.com/juanmmm21/mvcc-transaction-manager.git
cd mvcc-transaction-manager
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"  # or: pip install -e . pytest mypy ruff
```

## Usage

### CLI

```bash
# scripted walkthrough of every classic anomaly, per isolation level
python -m mvcc_transaction_manager demo

# interactive multi-transaction session
python -m mvcc_transaction_manager repl
begin serializable
put 1 alice on_call
commit 1
begin serializable
get 2 alice
...

# throughput under contention, per isolation level
python -m mvcc_transaction_manager benchmark --threads 8 --ops-per-thread 500 --keys 8
```

### Programmatic usage

```python
from mvcc_transaction_manager import (
    InMemoryRowStore,
    IsolationLevel,
    MVCCTransactionManager,
    RowId,
    SerializationConflictError,
)

mgr = MVCCTransactionManager(InMemoryRowStore())

setup = mgr.begin()
mgr.put(setup, RowId("alice"), b"on_call")
mgr.put(setup, RowId("bob"), b"on_call")
mgr.commit(setup)

t1 = mgr.begin(IsolationLevel.SERIALIZABLE)
t2 = mgr.begin(IsolationLevel.SERIALIZABLE)
mgr.get(t1, RowId("bob"))     # alice checks bob is still on call
mgr.get(t2, RowId("alice"))   # bob checks alice is still on call
mgr.put(t1, RowId("alice"), b"off_call")
mgr.put(t2, RowId("bob"), b"off_call")

mgr.commit(t1)
try:
    mgr.commit(t2)
except SerializationConflictError:
    mgr.abort(t2)  # write skew: one of the two must lose
```

## Data format / interface exposed to `nanosql`

The integration contract with the rest of the ecosystem is the `RowStore` `Protocol` in `protocols.py`:

```python
class RowStore(Protocol):
    def append_version(self, version: RowVersion) -> None: ...
    def versions_of(self, row_id: RowId) -> Sequence[RowVersion]: ...
    def row_ids(self) -> Iterator[RowId]: ...
    def prune_versions(self, row_id: RowId, keep: Sequence[RowVersion]) -> int: ...
```

Any storage engine that wants to sit underneath `MVCCTransactionManager` — including, inside `nanosql`, an adapter over `bplus-tree-storage-engine` or `lsm-tree-engine` — implements these four methods. `MVCCTransactionManager` never assumes anything about how versions are physically stored; it only ever compares `commit_seq` values.

## Development

```bash
pytest
ruff check .
ruff format --check .
mypy --strict src/
```

The test suite (42 tests) covers: the classic anomalies (dirty, non-repeatable and phantom read) reproduced and checked against the expected outcome at each isolation level; write-skew detection including a 3-transaction pivot scenario and a deliberate contrast showing `REPEATABLE_READ` does *not* catch it; real multi-threaded concurrency tests (unique id assignment under a thread storm, snapshot isolation observed from real OS threads, and a lost-update stress test with client-side retry); two independent property-based tests with a fixed seed — one comparing thousands of serial operations against a reference `dict`, the other comparing thousands of randomly interleaved multi-isolation-level operations against a hand-built commit-log visibility oracle; and CLI smoke tests.

## Benchmarks

Contention throughput measured with the `benchmark` subcommand (8 worker threads, 500 read-modify-write increments each, with client-side retry on conflict and a background `gc()` thread running throughout — without periodic GC, `SERIALIZABLE`'s conflict check degrades over a long run because it scans the retained transaction table, which otherwise grows unbounded; this was found by running the benchmark without it first):

**High contention — 8 shared rows:**

| Isolation level | Committed | Throughput | Conflicts | Retry rate |
|---|---|---|---|---|
| `read_committed` | 4000 | ~109,000 tx/s | 0 | 0.00% |
| `repeatable_read` | 4000 | ~92,000 tx/s | 0 | 0.00% |
| `serializable` | 4000 | ~12,000 tx/s | 34 | 0.85% |

**Low contention — 200 shared rows:**

| Isolation level | Committed | Throughput | Conflicts | Retry rate |
|---|---|---|---|---|
| `read_committed` | 4000 | ~311,000 tx/s | 0 | 0.00% |
| `repeatable_read` | 4000 | ~245,000 tx/s | 0 | 0.00% |
| `serializable` | 4000 | ~13,000 tx/s | 22 | 0.55% |

Two things stand out. First, `REPEATABLE_READ` is nearly as fast as `READ_COMMITTED` even under heavy contention: its only extra cost is a write-write check scoped to the committing transaction's own write set. Second, `SERIALIZABLE` is roughly an order of magnitude slower *even at a sub-1% retry rate* — that gap is not retries, it is the antidependency check itself, which scans every transaction still retained for conflict bookkeeping. This reference implementation trades that O(n) scan for simplicity; a production SSI implementation would index conflicts per row instead of per transaction to avoid it.

## Troubleshooting

- **`WriteConflictError` on `commit()`:** the transaction is `REPEATABLE_READ` or `SERIALIZABLE` and another transaction already committed a change to one of its written rows after its snapshot was taken. This is not a bug to work around — retry the transaction with a fresh snapshot (see the retry loop in the "Programmatic usage" example and in the benchmark).
- **`SerializationConflictError` on `commit()`:** the transaction is `SERIALIZABLE` and forms a write-skew cycle with a concurrent transaction. Same remedy: abort and retry with a fresh snapshot.
- **`TransactionNotActiveError`:** the transaction id is unknown, or the transaction already committed or aborted. Every transaction id is single-use.
- **A long-lived `REPEATABLE_READ`/`SERIALIZABLE` reader seems to prevent `gc()` from reclaiming anything:** that is by design — `gc()`'s horizon is the minimum snapshot among active transactions, so it never discards a version a currently open transaction might still need. Commit or abort it to let garbage collection proceed.

## License

MIT — see [`LICENSE`](./LICENSE).
