"""`scan_range`: idéntico a `scan()` filtrado por rango, sobre los tres
tipos de store (con `RangeRowStore`, sólo `BulkRowStore` y el `RowStore`
fila a fila), con escrituras propias sin confirmar, borrados y snapshots
antiguos; y el `read_set` sólo registra filas del rango."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

import pytest

from mvcc_transaction_manager import (
    InMemoryRowStore,
    IsolationLevel,
    MVCCTransactionManager,
    RangeRowStore,
    RowId,
    RowVersion,
)


class BulkOnlyStore:
    def __init__(self, inner: InMemoryRowStore) -> None:
        self._inner = inner

    def append_version(self, version: RowVersion) -> None:
        self._inner.append_version(version)

    def versions_of(self, row_id: RowId) -> Sequence[RowVersion]:
        return self._inner.versions_of(row_id)

    def row_ids(self) -> Iterator[RowId]:
        return self._inner.row_ids()

    def prune_versions(self, row_id: RowId, keep: Sequence[RowVersion]) -> int:
        return self._inner.prune_versions(row_id, keep)

    def all_versions(self) -> Iterator[RowVersion]:
        return self._inner.all_versions()


def _stores() -> list[object]:
    return [InMemoryRowStore(), BulkOnlyStore(InMemoryRowStore()), LegacyStore()]


class LegacyStore:
    def __init__(self) -> None:
        self._inner = InMemoryRowStore()

    def append_version(self, version: RowVersion) -> None:
        self._inner.append_version(version)

    def versions_of(self, row_id: RowId) -> Sequence[RowVersion]:
        return self._inner.versions_of(row_id)

    def row_ids(self) -> Iterator[RowId]:
        return self._inner.row_ids()

    def prune_versions(self, row_id: RowId, keep: Sequence[RowVersion]) -> int:
        return self._inner.prune_versions(row_id, keep)


def test_in_memory_store_implements_the_range_protocol() -> None:
    assert isinstance(InMemoryRowStore(), RangeRowStore)
    assert not isinstance(LegacyStore(), RangeRowStore)


@pytest.mark.parametrize("store_index", [0, 1, 2])
def test_scan_range_equals_filtered_scan(store_index: int) -> None:
    rng = random.Random(store_index)
    store = _stores()[store_index]
    manager = MVCCTransactionManager(store)  # type: ignore[arg-type]
    tables = ["a\x1f", "b\x1f", "ab\x1f"]
    snapshots = []
    for round_number in range(60):
        txn = manager.begin(IsolationLevel.REPEATABLE_READ)
        for _ in range(rng.randint(1, 4)):
            row = RowId(f"{rng.choice(tables)}{rng.randint(0, 9):03d}")
            if rng.random() < 0.25:
                manager.delete(txn, row)
            else:
                manager.put(txn, row, f"v{round_number}".encode())
        if rng.random() < 0.8:
            manager.commit(txn)
        else:
            snapshots.append(txn)  # transacción viva con escrituras propias
        if rng.random() < 0.2:
            snapshots.append(manager.begin(IsolationLevel.REPEATABLE_READ))
    for txn in snapshots:
        for prefix in tables:
            start, end = RowId(prefix), RowId(prefix[:-1] + chr(ord(prefix[-1]) + 1))
            expected = [(r, v) for r, v in manager.scan(txn) if start <= r < end]
            assert manager.scan_range(txn, start, end) == expected


def test_read_set_only_covers_the_range() -> None:
    manager = MVCCTransactionManager(InMemoryRowStore())
    setup = manager.begin()
    for row in ("a\x1f1", "b\x1f1"):
        manager.put(setup, RowId(row), b"x")
    manager.commit(setup)
    txn = manager.begin(IsolationLevel.SERIALIZABLE)
    manager.scan_range(txn, RowId("a\x1f"), RowId("a\x20"))
    read_set = manager._transactions[txn].read_set  # noqa: SLF001
    assert read_set == {RowId("a\x1f1")}
