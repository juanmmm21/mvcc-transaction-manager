"""Reproduce las anomalías clásicas de concurrencia y verifica que cada
nivel de aislamiento las evita (o las permite) según corresponde."""

from __future__ import annotations

import pytest

from mvcc_transaction_manager.models import (
    IsolationLevel,
    RowId,
    TransactionId,
    TransactionNotActiveError,
    WriteConflictError,
)
from mvcc_transaction_manager.pipeline import MVCCTransactionManager
from mvcc_transaction_manager.store import InMemoryRowStore

ALL_LEVELS = list(IsolationLevel)
SNAPSHOT_LEVELS = [IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE]


def _manager_with_committed_row(row_id: str, value: bytes) -> MVCCTransactionManager:
    mgr = MVCCTransactionManager(InMemoryRowStore())
    setup = mgr.begin()
    mgr.put(setup, RowId(row_id), value)
    mgr.commit(setup)
    return mgr


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_dirty_read_is_never_visible(level: IsolationLevel) -> None:
    mgr = _manager_with_committed_row("row", b"committed")

    writer = mgr.begin()
    mgr.put(writer, RowId("row"), b"uncommitted")

    reader = mgr.begin(level)
    assert mgr.get(reader, RowId("row")) == b"committed"

    mgr.abort(writer)
    mgr.commit(reader)


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_dirty_read_stays_hidden_even_if_writer_later_commits(level: IsolationLevel) -> None:
    mgr = _manager_with_committed_row("row", b"committed")

    writer = mgr.begin()
    mgr.put(writer, RowId("row"), b"in_flight")

    reader = mgr.begin(level)
    assert mgr.get(reader, RowId("row")) == b"committed"

    mgr.commit(writer)  # confirma después de que 'reader' ya haya leído
    if level is IsolationLevel.READ_COMMITTED:
        # RC toma un snapshot nuevo en cada lectura: una segunda lectura sí vería el commit
        assert mgr.get(reader, RowId("row")) == b"in_flight"
    else:
        # snapshot fijo: el commit posterior sigue sin ser visible
        assert mgr.get(reader, RowId("row")) == b"committed"


def test_read_committed_permits_non_repeatable_read() -> None:
    mgr = _manager_with_committed_row("row", b"A")

    reader = mgr.begin(IsolationLevel.READ_COMMITTED)
    first = mgr.get(reader, RowId("row"))

    other = mgr.begin()
    mgr.put(other, RowId("row"), b"B")
    mgr.commit(other)

    second = mgr.get(reader, RowId("row"))
    mgr.commit(reader)

    assert first == b"A"
    assert second == b"B"


@pytest.mark.parametrize("level", SNAPSHOT_LEVELS)
def test_snapshot_levels_prevent_non_repeatable_read(level: IsolationLevel) -> None:
    mgr = _manager_with_committed_row("row", b"A")

    reader = mgr.begin(level)
    first = mgr.get(reader, RowId("row"))

    other = mgr.begin()
    mgr.put(other, RowId("row"), b"B")
    mgr.commit(other)

    second = mgr.get(reader, RowId("row"))
    mgr.commit(reader)

    assert first == second == b"A"


def test_read_committed_permits_phantom_read() -> None:
    mgr = _manager_with_committed_row("row-1", b"x")

    reader = mgr.begin(IsolationLevel.READ_COMMITTED)
    first_scan = mgr.scan(reader)

    other = mgr.begin()
    mgr.put(other, RowId("row-2"), b"y")
    mgr.commit(other)

    second_scan = mgr.scan(reader)
    mgr.commit(reader)

    assert len(first_scan) == 1
    assert len(second_scan) == 2


@pytest.mark.parametrize("level", SNAPSHOT_LEVELS)
def test_snapshot_levels_prevent_phantom_read(level: IsolationLevel) -> None:
    mgr = _manager_with_committed_row("row-1", b"x")

    reader = mgr.begin(level)
    first_scan = mgr.scan(reader)

    other = mgr.begin()
    mgr.put(other, RowId("row-2"), b"y")
    mgr.commit(other)

    second_scan = mgr.scan(reader)
    mgr.commit(reader)

    assert first_scan == second_scan
    assert len(first_scan) == 1


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_read_your_own_writes(level: IsolationLevel) -> None:
    mgr = MVCCTransactionManager(InMemoryRowStore())
    txn = mgr.begin(level)
    assert mgr.get(txn, RowId("row")) is None
    mgr.put(txn, RowId("row"), b"local")
    assert mgr.get(txn, RowId("row")) == b"local"
    mgr.delete(txn, RowId("row"))
    assert mgr.get(txn, RowId("row")) is None
    mgr.commit(txn)


def test_read_committed_write_never_conflicts() -> None:
    mgr = _manager_with_committed_row("row", b"A")

    t1 = mgr.begin(IsolationLevel.READ_COMMITTED)
    t2 = mgr.begin(IsolationLevel.READ_COMMITTED)
    mgr.put(t1, RowId("row"), b"from_t1")
    mgr.put(t2, RowId("row"), b"from_t2")

    mgr.commit(t1)
    mgr.commit(t2)  # no debe lanzar: read committed no valida escrituras obsoletas
    assert mgr.get(mgr.begin(), RowId("row")) == b"from_t2"


@pytest.mark.parametrize("level", SNAPSHOT_LEVELS)
def test_snapshot_levels_abort_on_stale_write_write_conflict(level: IsolationLevel) -> None:
    mgr = _manager_with_committed_row("row", b"A")

    t1 = mgr.begin(level)
    t2 = mgr.begin(level)
    mgr.put(t1, RowId("row"), b"from_t1")
    mgr.put(t2, RowId("row"), b"from_t2")

    mgr.commit(t1)
    with pytest.raises(WriteConflictError):
        mgr.commit(t2)


def test_operating_on_finished_transaction_raises() -> None:
    mgr = MVCCTransactionManager(InMemoryRowStore())
    txn = mgr.begin()
    mgr.commit(txn)

    with pytest.raises(TransactionNotActiveError):
        mgr.get(txn, RowId("row"))
    with pytest.raises(TransactionNotActiveError):
        mgr.put(txn, RowId("row"), b"x")
    with pytest.raises(TransactionNotActiveError):
        mgr.commit(txn)
    with pytest.raises(TransactionNotActiveError):
        mgr.abort(txn)


def test_operating_on_unknown_transaction_raises() -> None:
    mgr = MVCCTransactionManager(InMemoryRowStore())
    with pytest.raises(TransactionNotActiveError):
        mgr.get(TransactionId(999), RowId("row"))
